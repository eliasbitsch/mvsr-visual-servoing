"""LocateAnything-3B in its own process: its remote code needs transformers 4.57, the rest of
the stack runs 5.x. Started by detectors/locate_anything.py with /opt/la-site (transformers
4.57.1) in front of PYTHONPATH; requests and answers are length-prefixed pickles on stdin/stdout.
"""
import pickle
import re
import struct
import sys

import numpy as np

# Model output: <box><x1><y1><x2><y2></box>, integers normalised to 0..1000 (model card)
_BOX = re.compile(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>')


def read_msg(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    return pickle.loads(stream.read(struct.unpack('<I', head)[0]))


def write_msg(stream, obj):
    data = pickle.dumps(obj)
    stream.write(struct.pack('<I', len(data)) + data)
    stream.flush()


def main():
    model_id, device = sys.argv[1], sys.argv[2]
    # stdout carries the protocol; library chatter goes to stderr
    out, sys.stdout = sys.stdout.buffer, sys.stderr
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True).to(device).eval()
    write_msg(out, 'ready')

    while (req := read_msg(sys.stdin.buffer)) is not None:
        image_rgb, prompt = req
        try:
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': Image.fromarray(image_rgb)},
                {'type': 'text', 'text': 'Locate all the instances that match the following '
                                         f'description: {prompt}.'}]}]
            text = processor.py_apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
            images, videos = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos,
                               return_tensors='pt').to(device)
            with torch.inference_mode():
                response = model.generate(
                    pixel_values=inputs['pixel_values'].to(torch.bfloat16),
                    input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                    image_grid_hws=inputs.get('image_grid_hws'), tokenizer=tokenizer,
                    max_new_tokens=64, use_cache=True, generation_mode='hybrid',
                    do_sample=False, verbose=False)
            answer = response[0] if isinstance(response, tuple) else response
            h, w = image_rgb.shape[:2]
            # Without magi_attention (falls back to SDPA) the model repeats its boxes until the
            # token limit: keep the first occurrence of each, in the order it named them
            boxes = []
            for m in dict.fromkeys(_BOX.findall(str(answer))):
                box = tuple(np.array(m, dtype=float) / 1000 * [w, h, w, h])
                if all(np.abs(np.subtract(box, b)).max() > 4 for b in boxes):
                    boxes.append(box)
            write_msg(out, ('ok', boxes))
        except Exception as exc:  # report, keep the model loaded
            write_msg(out, ('error', repr(exc)))


if __name__ == '__main__':
    main()
