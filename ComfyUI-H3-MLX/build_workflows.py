"""Build the checked-in user-facing ComfyUI workflows deterministically."""

from __future__ import annotations

import json
import uuid
from pathlib import Path


ROOT = Path(__file__).parent / "workflows"


def node(node_id, node_type, pos, size, title=None, widgets=None, inputs=None, outputs=None, color=None, bgcolor=None):
    value = {
        "id": node_id, "type": node_type, "pos": list(pos), "size": list(size),
        "flags": {}, "order": node_id, "mode": 0,
        "inputs": inputs or [], "outputs": outputs or [],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets or [],
    }
    if title:
        value["title"] = title
    if color:
        value["color"] = color
    if bgcolor:
        value["bgcolor"] = bgcolor
    return value


def link(link_id, origin, origin_slot, target, target_slot, value_type):
    return [link_id, origin, origin_slot, target, target_slot, value_type]


def build_t2v():
    nodes = [
        node(1, "MarkdownNote", (40, 30), (650, 180), "MiniMax H3 — Native MLX T2V", [
            "## MiniMax H3 — Native MLX T2V\nBase BF16 full attention on Apple Silicon. Text-only requests internally use the official first+last compatibility frames; no image input is required. Output contains H.264 video and synchronized stereo AAC audio."
        ], color="#243447", bgcolor="#172331"),
        node(2, "H3MLXModelLoader", (40, 270), (390, 100), "1 · Load native MLX model", ["MiniMax-H3"],
             outputs=[{"name":"h3_mlx_model","type":"H3_MLX_MODEL","links":[6]}]),
        node(3, "H3MLXTextConditioning", (470, 270), (620, 520), "2 · Raw prompt (passed unchanged)", [
            "Describe the scene, action, camera, dialogue, ambience, music, and sound effects."
        ], outputs=[{"name":"conditioning","type":"H3_MLX_CONDITIONING","links":[7]}]),
        node(4, "H3MLXResolution", (40, 450), (390, 180), "3 · Resolution / aspect ratio",
             ["16:9 landscape — 832×480", 832, 480],
             outputs=[{"name":"width","type":"INT","links":[1]}, {"name":"height","type":"INT","links":[2]}]),
        node(5, "H3MLXDuration", (40, 700), (390, 120), "4 · Duration (seconds)", [5.0],
             outputs=[{"name":"valid_frames","type":"INT","links":[3]}, {"name":"actual_seconds","type":"FLOAT","links":[]}]),
        node(6, "H3MLXGenerationSettings", (470, 680), (420, 360), "5 · Sampling settings", ["bfloat16","none","none","res_multistep","beta","mlx_fused_full_sdpa",20,0,"randomize"],
             inputs=[{"name":"width","type":"INT","link":1},{"name":"height","type":"INT","link":2},{"name":"frames","type":"INT","link":3}],
             outputs=[{"name":"generation_settings","type":"H3_MLX_SETTINGS","links":[8]}]),
        node(7, "H3MLXGenerate", (1160, 350), (430, 180), "6 · Native MLX sample + AV decode", [],
             inputs=[{"name":"h3_mlx_model","type":"H3_MLX_MODEL","link":6},{"name":"conditioning","type":"H3_MLX_CONDITIONING","link":7},{"name":"generation_settings","type":"H3_MLX_SETTINGS","link":8}],
             outputs=[{"name":"video","type":"VIDEO","links":[9]}]),
        node(8, "H3MLXOutput", (1660, 350), (430, 150), "7 · Preview / save audiovisual MP4", ["h3_mlx/t2v/H3"],
             inputs=[{"name":"video","type":"VIDEO","link":9}],
             outputs=[{"name":"mp4_path","type":"STRING","links":[]},{"name":"diagnostics_path","type":"STRING","links":[]},{"name":"step_timings","type":"STRING","links":[]}]),
        node(9, "MarkdownNote", (1160, 610), (930, 220), "Usage", [
            "### Controls\n- The prompt is passed to the tokenizer byte-for-byte as entered; this workflow does not rewrite it.\n- T2V compatibility frames are fixed internal conditioning inputs and are not user prompt content.\n- **Duration:** automatically snaps to the nearest valid `17n+5` frame count at 24 fps.\n- **Steps:** production Base uses 20.\n- **Seed:** supports ComfyUI randomize-after-run behavior.\n\nNo Torch latent or frame is created in the generation path."
        ], color="#303030", bgcolor="#202020"),
    ]
    links = [link(1,4,0,6,0,"INT"),link(2,4,1,6,1,"INT"),link(3,5,0,6,2,"INT"),link(6,2,0,7,0,"H3_MLX_MODEL"),link(7,3,0,7,1,"H3_MLX_CONDITIONING"),link(8,6,0,7,2,"H3_MLX_SETTINGS"),link(9,7,0,8,0,"VIDEO")]
    groups = [
        {"id":1,"title":"INPUT / MODEL","bounding":[20,240,430,610],"color":"#3f789e","font_size":24,"flags":{}},
        {"id":2,"title":"PROMPT / CONDITIONING","bounding":[450,240,660,390],"color":"#3f789e","font_size":24,"flags":{}},
        {"id":3,"title":"SAMPLING","bounding":[450,650,460,310],"color":"#8b6f47","font_size":24,"flags":{}},
        {"id":4,"title":"NATIVE MLX GENERATION","bounding":[1140,320,470,240],"color":"#5d7f55","font_size":24,"flags":{}},
        {"id":5,"title":"VIDEO + AUDIO OUTPUT","bounding":[1640,320,470,240],"color":"#8a5064","font_size":24,"flags":{}},
    ]
    return {"id":str(uuid.uuid4()),"revision":0,"last_node_id":9,"last_link_id":9,"nodes":nodes,"links":links,"groups":groups,"config":{},"extra":{"ds":{"scale":0.72,"offset":[130,80]}},"version":0.4}


def build_i2v():
    nodes = [
        node(1, "MarkdownNote", (20, 20), (760, 190), "MiniMax H3 — Native MLX I2V / FL2V", [
            "## First / last frame workflow\nReplace the images below. Keep only the first-frame link for I2V, only the last-frame link for L2V, or both links for FL2V. The images are encoded by the native MLX Visual VAE and Qwen3-VL layer-50 conditioner."
        ]),
        node(2, "H3MLXModelLoader", (20, 250), (380, 100), widgets=["MiniMax-H3"],
             outputs=[{"name":"h3_mlx_model","type":"H3_MLX_MODEL","links":[9,12]}]),
        node(3, "H3MLXTextConditioning", (440, 250), (620, 500), "Raw prompt (passed unchanged)", [
            "Continue from <Picture 1>. Describe the target motion, environment, camera, dialogue, ambience, music, and sound effects."
        ], outputs=[{"name":"conditioning","type":"H3_MLX_CONDITIONING","links":[10]}]),
        node(4, "LoadImage", (20, 410), (380, 310), "First frame (required for I2V)", ["DragonballZ.png", "image"],
             outputs=[{"name":"IMAGE","type":"IMAGE","links":[7]},{"name":"MASK","type":"MASK","links":[]}]),
        node(5, "LoadImage", (20, 760), (380, 310), "Last frame (optional — connect for FL2V)", ["DragonballZ.png", "image"],
             outputs=[{"name":"IMAGE","type":"IMAGE","links":[]},{"name":"MASK","type":"MASK","links":[]}]),
        node(6, "H3MLXResolution", (440, 800), (380, 170), widgets=["16:9 landscape — 832×480",832,480],
             outputs=[{"name":"width","type":"INT","links":[1]},{"name":"height","type":"INT","links":[2]}]),
        node(7, "H3MLXDuration", (440, 1010), (380, 110), widgets=[5.0],
             outputs=[{"name":"valid_frames","type":"INT","links":[3]},{"name":"actual_seconds","type":"FLOAT","links":[]}]),
        node(8, "H3MLXGenerationSettings", (860, 830), (390, 330), widgets=["bfloat16","none","none","res_multistep","beta","mlx_fused_full_sdpa",20,0,"randomize"],
             inputs=[{"name":"width","type":"INT","link":1},{"name":"height","type":"INT","link":2},{"name":"frames","type":"INT","link":3}],
             outputs=[{"name":"generation_settings","type":"H3_MLX_SETTINGS","links":[11]}]),
        node(9, "H3MLXKeyframeSample", (1320, 340), (470, 260), "Native MLX T2V / I2V / FL2V sample", [],
             inputs=[{"name":"h3_mlx_model","type":"H3_MLX_MODEL","link":9},
                     {"name":"conditioning","type":"H3_MLX_CONDITIONING","link":10},
                     {"name":"generation_settings","type":"H3_MLX_SETTINGS","link":11},
                     {"name":"first_frame","type":"IMAGE","link":7},
                     {"name":"last_frame","type":"IMAGE","link":None}],
             outputs=[{"name":"latent","type":"H3_MLX_LATENT","links":[13]},{"name":"mode","type":"STRING","links":[]}]),
        node(10, "H3MLXVAEDecode", (1840, 340), (380, 150), inputs=[
            {"name":"h3_mlx_model","type":"H3_MLX_MODEL","link":12},{"name":"latent","type":"H3_MLX_LATENT","link":13}],
             outputs=[{"name":"video","type":"VIDEO","links":[14]}]),
        node(11, "H3MLXOutput", (2270, 340), (410, 150), widgets=["h3_mlx/i2v/H3"],
             inputs=[{"name":"video","type":"VIDEO","link":14}],
             outputs=[{"name":"mp4_path","type":"STRING","links":[]},{"name":"diagnostics_path","type":"STRING","links":[]},{"name":"step_timings","type":"STRING","links":[]}]),
    ]
    links = [link(1,6,0,8,0,"INT"),link(2,6,1,8,1,"INT"),link(3,7,0,8,2,"INT"),
             link(7,4,0,9,3,"IMAGE"),link(9,2,0,9,0,"H3_MLX_MODEL"),
             link(10,3,0,9,1,"H3_MLX_CONDITIONING"),link(11,8,0,9,2,"H3_MLX_SETTINGS"),
             link(12,2,0,10,0,"H3_MLX_MODEL"),link(13,9,0,10,1,"H3_MLX_LATENT"),link(14,10,0,11,0,"VIDEO")]
    return {"id":str(uuid.uuid4()),"revision":0,"last_node_id":11,"last_link_id":14,"nodes":nodes,"links":links,"groups":[],"config":{},"extra":{"ds":{"scale":0.65,"offset":[80,80]}},"version":0.4}


def build_r2v():
    reference_nodes = [
        node(3 + index, "LoadImage", (20, 400 + (index - 1) * 330), (400, 290),
             f"Reference image {index}" + (" (required)" if index == 1 else " (optional)"),
             ["DragonballZ.png", "image"],
             outputs=[{"name":"IMAGE","type":"IMAGE","links":[4] if index == 1 else []},
                      {"name":"MASK","type":"MASK","links":[]}])
        for index in range(1, 10)
    ]
    nodes = [
        node(1, "MarkdownNote", (20, 20), (900, 190), "MiniMax H3 — Native MLX Ref2VA", [
            "## Image-reference Ref2VA — Base BF16\nPaste the complete official Ref2VA prompt into the single raw-prompt box; it is passed unchanged. Connect 1–9 images in Picture order. Native MLX reference-video and reference-audio conditioning are unsupported and are not silently replaced with image or FL2VA conditioning."
        ]),
        node(2, "H3MLXRefModelLoader", (20, 250), (400, 110), "1 · Load Ref2VA partition", ["MiniMax-H3"],
             outputs=[{"name":"h3_mlx_model","type":"H3_MLX_MODEL","links":[5,8]}]),
        *reference_nodes,
        node(13, "H3MLXTextConditioning", (460, 250), (760, 650), "2 · Complete official Ref2VA prompt (raw / unchanged)", [
            "subject_definitions:\n<Subject 1> is the person or principal subject shown in <Picture 1>.\n\nsummary:\n[reference generation] Create a new cinematic shot preserving <Subject 1>'s identity, appearance, clothing, and visual texture.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - preserve the identity and appearance defined by <Picture 1>.\n\ndetailed_description:\n[Shot 1] <Picture 1> is fully referenced for subject identity and appearance. Describe the target action, environment, camera movement, and timing here.\n\noverall_soundscape:\nNatural synchronized ambience and physical sound effects matching the target action.\n\nnon_diegetic_music:\nN/A"
        ], outputs=[{"name":"conditioning","type":"H3_MLX_CONDITIONING","links":[6]}]),
        node(14, "H3MLXResolution", (470, 940), (390, 180), "3 · Target resolution", ["16:9 landscape — 832×480",832,480],
             outputs=[{"name":"width","type":"INT","links":[1]},{"name":"height","type":"INT","links":[2]}]),
        node(15, "H3MLXDuration", (470, 1160), (390, 120), "4 · Duration", [5.0],
             outputs=[{"name":"valid_frames","type":"INT","links":[3]},{"name":"actual_seconds","type":"FLOAT","links":[]}]),
        node(16, "H3MLXGenerationSettings", (900, 940), (410, 340), "5 · Sampling settings", ["bfloat16","none","none","res_multistep","beta","mlx_fused_full_sdpa",20,0,"randomize"],
             inputs=[{"name":"width","type":"INT","link":1},{"name":"height","type":"INT","link":2},{"name":"frames","type":"INT","link":3}],
             outputs=[{"name":"generation_settings","type":"H3_MLX_SETTINGS","links":[7]}]),
        node(17, "H3MLXReferenceSample", (1370, 350), (520, 430), "6 · Native Ref2VA image sample", ["match"],
             inputs=[{"name":"h3_mlx_model","type":"H3_MLX_MODEL","link":5},
                     {"name":"conditioning","type":"H3_MLX_CONDITIONING","link":6},
                     {"name":"generation_settings","type":"H3_MLX_SETTINGS","link":7},
                     *[{"name":f"reference_image_{index}","type":"IMAGE","link":4 if index == 1 else None}
                       for index in range(1, 10)]],
             outputs=[{"name":"latent","type":"H3_MLX_LATENT","links":[9]},{"name":"mode","type":"STRING","links":[]}]),
        node(18, "H3MLXVAEDecode", (1950, 350), (390, 160), "7 · Native AV decode", [],
             inputs=[{"name":"h3_mlx_model","type":"H3_MLX_MODEL","link":8},{"name":"latent","type":"H3_MLX_LATENT","link":9}],
             outputs=[{"name":"video","type":"VIDEO","links":[10]}]),
        node(19, "H3MLXOutput", (2390, 350), (430, 160), "8 · Preview / save audiovisual MP4", ["h3_mlx/r2v/H3"],
             inputs=[{"name":"video","type":"VIDEO","link":10}],
             outputs=[{"name":"mp4_path","type":"STRING","links":[]},{"name":"diagnostics_path","type":"STRING","links":[]},{"name":"step_timings","type":"STRING","links":[]}]),
    ]
    links = [link(1,14,0,16,0,"INT"),link(2,14,1,16,1,"INT"),link(3,15,0,16,2,"INT"),
             link(4,4,0,17,3,"IMAGE"),link(5,2,0,17,0,"H3_MLX_MODEL"),
             link(6,13,0,17,1,"H3_MLX_CONDITIONING"),link(7,16,0,17,2,"H3_MLX_SETTINGS"),
             link(8,2,0,18,0,"H3_MLX_MODEL"),link(9,17,0,18,1,"H3_MLX_LATENT"),
             link(10,18,0,19,0,"VIDEO")]
    return {"id":str(uuid.uuid4()),"revision":0,"last_node_id":19,"last_link_id":10,"nodes":nodes,"links":links,"groups":[],"config":{},"extra":{"ds":{"scale":0.48,"offset":[80,80]}},"version":0.4}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "h3_mlx_t2v.json").write_text(json.dumps(build_t2v(), indent=2, ensure_ascii=False) + "\n")
    (ROOT / "h3_mlx_i2v.json").write_text(json.dumps(build_i2v(), indent=2, ensure_ascii=False) + "\n")
    (ROOT / "h3_mlx_r2v.json").write_text(json.dumps(build_r2v(), indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
