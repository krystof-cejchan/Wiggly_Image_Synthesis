"""Inline every {{IMG:name}} placeholder as a base64 data URI (artifacts must be self-contained)."""
import base64, os, re, sys
SC = "/tmp/claude-1001/-home-krystof-prg-Wiggly-Image-Synthesis/94a0a51f-8d0c-4334-9c62-454de430eec2/scratchpad"
FIGS = f"{SC}/figs"
ALT = {
 "toy_field_flow": "Left: learned velocity arrows pointing toward a wiggly ribbon. Right: noise particles flowing along those arrows and landing on the ribbon.",
 "toy_conditional": "Straight amber lines connecting scattered grey noise points to teal data points on a sine-shaped ribbon.",
 "toy_marginal": "Many thin amber arrows fanning out from one point in different directions, with a thick plum arrow showing their average.",
 "straight_vs_curved": "A straight teal line and a curved dashed amber line joining the same two endpoints.",
}
def alt(n):
    if n in ALT: return ALT[n]
    if n.startswith("interp_"):
        return f"Microtubule crop linearly blended with Gaussian noise at t={int(n.split('_')[1])/100:g}."
    if n.startswith("traj_"):
        p = n.split("_"); ph = "5.8" if p[1] == "low" else "8.8"
        return f"Generated sample at pH {ph}, ODE integrated to t={int(p[2])/100:g}."
    return n

src = open(f"{SC}/study_template.html").read()
missing = []
def sub(m):
    name = m.group(1)
    path = f"{FIGS}/{name}.png"
    if not os.path.exists(path):
        missing.append(name); return ""
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    cls = "plot" if name.startswith(("toy_", "straight_")) else ""
    return f'<img{" class=\"" + cls + "\"" if cls else ""} src="data:image/png;base64,{b64}" alt="{alt(name)}">'

out = re.sub(r"\{\{IMG:([a-z0-9_]+)\}\}", sub, src)
if missing:
    print("MISSING:", missing); sys.exit(1)
if "{{" in out:
    print("unreplaced placeholder remains"); sys.exit(1)
dest = f"{SC}/flow-matching-study.html"
open(dest, "w").write(out)
print(f"wrote {dest}  ({len(out)/1024/1024:.2f} MB)")
