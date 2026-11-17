"""Flowchart of the pipeline for the demo app (Mermaid, standard flowchart symbols): terminal,
input/output, predefined process, process, decision, data store. Active stages are tinted."""

THEME = ("%%{init: {'theme': 'base', 'themeVariables': {"
         "'fontFamily': 'Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif', "
         "'fontSize': '15px', 'primaryColor': '#ffffff', 'primaryBorderColor': '#48484a', "
         "'primaryTextColor': '#1d1d1f', 'lineColor': '#8e8e93', "
         "'edgeLabelBackground': '#ffffff', 'tertiaryColor': '#ffffff'}, "
         "'flowchart': {'curve': 'basis', 'nodeSpacing': 34, 'rankSpacing': 46, "
         "'padding': 14, 'useMaxWidth': true}}}%%")

STYLES = [
    'classDef on fill:#e9f7ee,stroke:#1f8a4c,stroke-width:2px,color:#0f3d22',
    'classDef off fill:#ffffff,stroke:#8e8e93,stroke-width:1.2px,color:#3a3a3c',
    'classDef eval fill:#fffaf0,stroke:#c9a227,stroke-width:1.2px,stroke-dasharray:5 4,color:#6b5310',
]


def mermaid(s):
    """s: live values and flags collected by Demo.flow()."""
    def q(text):  # Mermaid label, line breaks as <br/>
        return '"' + text.replace('"', "'") + '"'

    lines = [THEME, 'flowchart LR']
    lines += [
        f'  P([{q("Text prompt<br/><i>" + s["prompt"] + "</i>")}])',
        f'  CAM[/{q("Wrist camera<br/>" + s["cam"])}/]',
        f'  D1{{{q("Target<br/>tracked?")}}}',
        f'  FM[[{q("Foundation model<br/><b>" + s["detector"] + "</b><br/>" + s["latency"])}]]',
        f'  TR[{q("Tracker<br/><b>" + s["tracker"] + "</b>")}]',
        f'  FE[{q("Image features<br/>" + s["features"])}]',
        f'  IB[{q("IBVS control law<br/>" + s["error"])}]',
        f'  D2{{{q("Converged?")}}}',
        f'  KI[{q("Hand-eye + Jacobian<br/>joint velocities")}]',
        f'  RO[/{q("UR5e in Gazebo<br/>" + s["speed"])}/]',
        f'  HO([{q("Hold pose")}])',
        f'  GT[({q("Ground truth<br/>evaluation only")})]',
        '  P --> FM',
        '  CAM -- "/camera/image_raw" --> D1',
        '  D1 -- no --> FM',
        '  FM -- "box / mask" --> TR',
        '  D1 -- yes --> TR',
        '  TR -- mask --> FE',
        '  FE -- "/vs/features" --> IB',
        '  IB --> D2',
        '  D2 -- yes --> HO',
        '  D2 -- no --> KI',
        '  KI -- "/forward_velocity_controller" --> RO',
        '  RO -. "moves the camera" .-> CAM',
        '  RO -.-> GT',
    ]
    lines += STYLES
    for node in ['P', 'CAM', 'D1', 'FM', 'TR', 'FE', 'IB', 'D2', 'KI', 'RO', 'HO']:
        lines.append(f'  class {node} {"on" if s["on"].get(node) else "off"}')
    lines.append('  class GT eval')
    return '\n'.join(lines)
