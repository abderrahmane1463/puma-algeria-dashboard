import pathlib

p = pathlib.Path('app.py')
txt = p.read_text(encoding='utf-8')
txt = txt.replace("use_container_width=True", "width='stretch'")
p.write_text(txt, encoding='utf-8')
print("Done - replaced all instances")
