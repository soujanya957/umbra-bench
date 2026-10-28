"""SDS positive control, behind the SDS card on the work log (printout saved as
results/fastshadow/research/sds_control_output.txt).

Positive control: on images SD 1.5 drew itself, does the same measurement separate prompts?"""
import os, sys
import numpy as np, torch
UB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(UB, "results", "fastshadow", "research")  # images + printout read by make_research_notes.py
sys.path.insert(0, UB); sys.path.insert(0, os.path.join(UB, "scripts"))
from sds_score import SDSScorer
sc = SDSScorer("sd15", n_noise=4, batch=8)
gens = {"dog": "a photo of a dog on grass", "letter A": 'the capital letter "A", black on a white background',
        "letter O": 'the capital letter "O", black on a white background'}
tests = {"dog": "a photo of a dog on grass", "cat": "a photo of a cat on a sofa",
         "letter A": 'the capital letter "A", black on a white background', "letter O": 'the capital letter "O", black on a white background',
         "letter X": 'the capital letter "X", black on a white background', "empty": ""}
g = torch.Generator("cuda").manual_seed(1)
for name, p in gens.items():
    with torch.no_grad():
        img = sc.pipe(p, num_inference_steps=25, generator=g, output_type="pt").images  # (1,3,512,512) in [0,1]
        z = (sc.vae.encode((img * 2 - 1).half()).latent_dist.mean * sc.vae.config.scaling_factor).half()
    E = sc.errors(z, list(tests.values()))[0]
    best = min(range(len(E)), key=lambda i: E[i])
    print(f"image drawn from '{name}': " + "  ".join(f"{k} {e:.5f}" for k, e in zip(tests, E)) + f"   -> best: {list(tests)[best]}")
    import PIL.Image
    PIL.Image.fromarray((img[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)).save(
        os.path.join(OUT, f"control_{name.replace(' ', '_')}.png"))
