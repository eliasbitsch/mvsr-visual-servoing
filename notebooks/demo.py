import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Visual Servoing live demo", css_file="demo.css")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from vs_evaluation.bench_detectors import PROMPTS
    from vs_evaluation.demo import Demo
    from vs_evaluation.run_eval import TARGETS
    return Demo, PROMPTS, TARGETS, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Visual Servoing with Foundation Models: live demo
    Group C (Bitsch, Stix, Zhou), track FM. Lichtblick shows robot and camera next to it:
    [localhost:8081](http://localhost:8081/?ds=foxglove-websocket&ds.url=ws://localhost:8766)

    **Text prompt → foundation model finds the target → SAM 2 tracks it between model calls →
    image features (centre, area, orientation) → IBVS → joint velocities.**
    The simulation's ground truth only feeds the numbers shown here, never the controller.
    """)
    return


@app.cell
def _(Demo, mo):
    demo = Demo()
    # Bumped after every action, so status card and session table refresh
    get_tick, set_tick = mo.state(0)
    return demo, get_tick, set_tick


@app.cell
def _(demo, get_tick, mo):
    get_tick()
    _last = demo.history[-1] if demo.history else None
    _result = ('-' if _last is None else
               f'{"✅" if _last["success"] else "❌"} {_last["pos error [mm]"]} mm')
    mo.vstack([
        mo.md('## Status'),
        mo.hstack([
            mo.stat(demo.params['detector'], label='foundation model', bordered=True),
            mo.stat(demo.params['tracker'], label='tracker', bordered=True),
            mo.stat(f'"{demo.text}"' if demo.text else '-', label='prompt in use', bordered=True),
            mo.stat(demo.target or '-', label='target', bordered=True),
        ], widths='equal'),
        mo.hstack([
            mo.stat(demo.start_name or '-', label='start configuration', bordered=True),
            mo.stat(demo.disturbance, label='disturbance', bordered=True),
            mo.stat(_result, label='last servo run', bordered=True),
            mo.stat(f'{sum(r["success"] for r in demo.history)}/{len(demo.history)}',
                    label='successful runs', bordered=True),
        ], widths='equal'),
    ])
    return


@app.cell
def _(mo):
    flow_refresh = mo.ui.refresh(options=['1s', '2s', '5s'], default_interval='5s',
                                 label='live pipeline')
    return (flow_refresh,)


@app.cell
def _(demo, flow_refresh, mo):
    flow_refresh.value
    demo.spin_for(0.3)  # take in the latest messages
    mo.vstack([
        mo.hstack([mo.md('## Pipeline, live'), flow_refresh],
                  justify='space-between'),
        demo.flow(),
        mo.md('*During a servo run the diagram updates in the run\'s output below.*'),
    ])
    return


@app.cell
def _(TARGETS, mo):
    target = mo.ui.dropdown(options=list(TARGETS), value='blue_box', label='target')
    start = mo.ui.dropdown(
        options={'S1 translation': 1, 'S2 rotation 30 deg': 2, 'S3 combined': 3,
                 'S4 far (+20 cm)': 4, 'S5 tilted': 5, 'S6 rotation 90 deg': 6,
                 'S7 rotation 170 deg': 7, 'S8 close (-5 cm)': 8},
        value='S2 rotation 30 deg', label='start configuration')
    return start, target


@app.cell
def _(PROMPTS, mo, target):
    # Recreated when the target changes; a typed prompt wins over the preset
    _levels = PROMPTS[target.value]
    prompt_preset = mo.ui.dropdown(
        options={f'{_lvl}: {_txt}': _txt for _lvl, _txt in _levels.items()},
        value=f'attribute: {_levels["attribute"]}', label='prompt preset')
    prompt_custom = mo.ui.text(placeholder='... or type your own prompt', full_width=True)
    return prompt_custom, prompt_preset


@app.cell
def _(prompt_custom, prompt_preset):
    servo_prompt = prompt_custom.value.strip() or prompt_preset.value
    return (servo_prompt,)


@app.cell
def _(mo):
    btn_reference = mo.ui.run_button(label='1 reference pose + detect')
    btn_capture = mo.ui.run_button(label='2 capture reference')
    btn_start = mo.ui.run_button(label='3 go to start')
    btn_servo = mo.ui.run_button(label='4 servo', kind='success')
    btn_lost = mo.ui.run_button(label='safe-stop test', kind='danger')
    occlusion = mo.ui.slider(0, 0.6, step=0.1, value=0.3, label='occlusion', show_value=True)
    light = mo.ui.slider(0.1, 1.5, step=0.1, value=0.3, label='light', show_value=True)
    btn_occlude = mo.ui.run_button(label='add occluder')
    btn_light = mo.ui.run_button(label='set light')
    btn_clear = mo.ui.run_button(label='clear disturbances')
    gz_gui = mo.ui.switch(value=False, label='Gazebo GUI window (the lab world; costs GPU)')
    return (btn_capture, btn_clear, btn_light, btn_lost, btn_occlude, btn_reference,
            btn_servo, btn_start, gz_gui, light, occlusion)


@app.cell
def _(demo, gz_gui):
    demo.gazebo_gui(gz_gui.value)
    return


@app.cell
def _(btn_capture, btn_clear, btn_light, btn_lost, btn_occlude, btn_reference, btn_servo,
      btn_start, gz_gui, light, mo, occlusion, prompt_custom, prompt_preset, servo_prompt, start,
      target):
    mo.vstack([
        mo.hstack([mo.md('## Pipeline, step by step'), gz_gui], justify='space-between'),
        mo.hstack([target, start, prompt_preset], justify='start'),
        prompt_custom,
        mo.md(f'Prompt for the next reference: **"{servo_prompt}"**'),
        mo.hstack([btn_reference, btn_capture, btn_start, btn_servo, btn_lost], justify='start'),
        mo.md('**Disturbances**'),
        mo.hstack([occlusion, btn_occlude, light, btn_light, btn_clear], justify='start'),
    ])
    return


@app.cell
def _(btn_reference, demo, mo, servo_prompt, set_tick, target):
    mo.stop(not btn_reference.value, mo.md('*1: drive to the reference pose, detect the target*'))
    demo.reference(target.value, prompt=servo_prompt)
    set_tick(lambda t: t + 1)
    return


@app.cell
def _(btn_capture, demo, mo):
    mo.stop(not btn_capture.value, mo.md('*2: the current features become the desired ones*'))
    demo.capture_reference()
    return


@app.cell
def _(btn_start, demo, mo, set_tick, start):
    mo.stop(not btn_start.value, mo.md("*3: one of the lab's start configurations*"))
    demo.go_start(start.value)
    set_tick(lambda t: t + 1)
    return


@app.cell
def _(btn_servo, demo, mo, set_tick):
    mo.stop(not btn_servo.value, mo.md('*4: close the loop (live: feature error, camera velocity, '
                                       'ground-truth position error)*'))
    demo.servo()
    demo.trajectories()
    set_tick(lambda t: t + 1)
    return


@app.cell
def _(btn_lost, demo, mo):
    mo.stop(not btn_lost.value, mo.md('*Safe stop: the target leaves the view 3 s into the run*'))
    demo.servo(lose_after=3)
    return


@app.cell
def _(btn_clear, btn_light, btn_occlude, demo, light, mo, occlusion, set_tick):
    mo.stop(not (btn_occlude.value or btn_light.value or btn_clear.value))
    if btn_clear.value:
        demo.clear()
    if btn_occlude.value:
        demo.occlude(occlusion.value)
    if btn_light.value:
        demo.light(light.value)
    set_tick(lambda t: t + 1)
    return


@app.cell
def _(demo, get_tick, mo, pd):
    get_tick()
    mo.stop(not demo.history, mo.md('*Session: every servo run of this demo appears here*'))
    _h = pd.DataFrame(demo.history)
    mo.vstack([
        mo.md(f'## Servoing quality, this session: {int(_h.success.sum())}/{len(_h)} within '
              f'5 mm / 2 deg'),
        _h,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Prompts
    Same view, different wording (one per line). Objects that are not in the scene show false
    positives. The servo prompt is set again afterwards.
    """)
    return


@app.cell
def _(mo):
    prompt_list = mo.ui.text_area(
        value='box\nblue box\nblue carton\nthe flat rectangular object\nyellow sphere\npurple bottle',
        rows=6, label='prompt list')
    btn_compare = mo.ui.run_button(label='compare prompts')
    free_prompt = mo.ui.text(value='the object between the can and the ball', label='free prompt',
                             full_width=True)
    btn_free = mo.ui.run_button(label='try it')
    mo.hstack([mo.vstack([prompt_list, btn_compare]), mo.vstack([free_prompt, btn_free])],
              widths=[1, 1])
    return btn_compare, btn_free, free_prompt, prompt_list


@app.cell
def _(btn_compare, demo, mo, prompt_list):
    mo.stop(not btn_compare.value)
    demo.compare_prompts([_p.strip() for _p in prompt_list.value.splitlines() if _p.strip()])
    return


@app.cell
def _(btn_free, demo, free_prompt, mo):
    mo.stop(not btn_free.value)
    demo.prompt(free_prompt.value)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Results of the automated evaluation
    Unattended runs over 4 targets × 8 start configurations (`scripts/run_campaign.py`).
    **Visual servoing:** success = converged and within 5 mm / 2 deg of the reference pose
    (ground truth); the variable is the tracker that bridges the model calls.
    """)
    return


@app.cell
def _(pd):
    _runs = {'DINO + SAM 2': '20261112_235523_full_dino_sam2',
             'DINO + CSRT': '20261113_000850_full_dino_csrt'}
    _rows = []
    for _name, _folder in _runs.items():
        _df = pd.read_csv(f'/root/vs_data/eval/{_folder}/summary.csv')
        for _target, _g in _df.groupby('target'):
            _rows.append({'config': _name, 'target': _target,
                          'success': f'{_g.success.sum()}/{len(_g)}',
                          'median error [mm]': round(_g.final_pos_err_m.median() * 1000, 2),
                          'median time [s]': round(_g.conv_time_s.median(), 1)})
    pd.DataFrame(_rows).pivot(index='target', columns='config',
                              values=['success', 'median error [mm]', 'median time [s]'])
    return


@app.cell
def _(mo, pd):
    _d = pd.read_csv('/root/vs_data/eval/20261112_235523_full_dino_sam2/summary.csv')
    _p = _d.pivot_table(index='target', columns='start', values='success', aggfunc='max')
    _p.columns = [f'S{_c}' for _c in _p.columns]
    mo.vstack([
        mo.md('Per start configuration (DINO + SAM 2, 1 = success):'),
        _p.reset_index(),
    ])
    return


@app.cell
def _(mo, pd):
    _b = pd.read_csv('/root/vs_data/eval/final_smoke/bench/results.csv')
    _pos = _b[_b.level != 'negative']
    _t = _pos.pivot_table(index='detector', columns='level', values='correct', aggfunc='mean')
    _t['false positives'] = _b[_b.level == 'negative'].groupby('detector').false_positive.mean()
    _t['median latency [s]'] = _b.groupby('detector').latency_s.median()
    mo.vstack([
        mo.md('**Foundation-model part** (prompt and model comparison of the FM track): '
              'localisation offline, first smoke test with few frames, indicative only. '
              'Correct = IoU ≥ 0.5 with the projected ground-truth box.'),
        _t.round(2).sort_values('attribute', ascending=False).reset_index(),
    ])
    return


if __name__ == "__main__":
    app.run()
