"""CPU extraction with pinned checkpoints and the study's exact preprocessing.

Author: James Edward Ball. Encoders remain frozen; no remote model code executes.
"""

import calendar
import gc
import importlib.util
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import PACKAGE, download, read


def normalise(x, mean, std):
    return (x * np.float32(10000) - np.asarray(mean, np.float32)[:, None, None]) / np.asarray(
        std, np.float32
    )[:, None, None]


WEIGHTS = {
    "clay": (
        "https://huggingface.co/made-with-clay/Clay/resolve/70200ebcccdf67bf2a0cb9984c77ddee26c10ed2/v1.5/clay-v1.5.ckpt",
        "21432069250b9b3f9a65ffd0071c5ad56b793247285ab0604edf7f531d4798d0",
    ),
    "prithvi": (
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/9eb1b1102806593963daa333bcc491b1c6f8562f/Prithvi_EO_V2_300M.pt",
        "faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7",
    ),
    "remoteclip": (
        "https://huggingface.co/chendelong/RemoteCLIP/resolve/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt",
        "60014e395d930a3f2963d1d89c8522bf4ad56775571e4356e866864789af85c4",
    ),
}


def encoder(name, work, phrases):
    import torch
    import yaml

    from clay_probe.vendor.model import Encoder

    random = name.startswith("random_")
    arch = name.split("_")[1] if random else name
    if random:
        torch.manual_seed(int(name.split("_")[-1]))
    weight = None
    if not random:
        key = (
            "remoteclip" if name.startswith("remoteclip") else "clay" if name == "teacher" else name
        )
        url, h = WEIGHTS[key]
        weight = download(url, work / "weights" / (key + ".pt"), h)
    text = None
    if arch == "clay":
        kwargs = dict(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4,
        )
        if random:
            model = Encoder(**kwargs)
        else:
            with torch.device("meta"):
                model = Encoder(**kwargs)
            state = torch.load(weight, weights_only=True, mmap=True, map_location="cpu")[
                "state_dict"
            ]
            prefix = "model.encoder."
            model.load_state_dict(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
                strict=True,
                assign=True,
            )
        sensor = yaml.safe_load((PACKAGE / "src/clay_probe/vendor/metadata.yaml").read_text())[
            "sentinel-2-l2a"
        ]["bands"]
        bands = ["blue", "green", "red", "nir08", "swir16", "swir22"]

        def forward(x):
            raw = normalise(
                x, [sensor["mean"][b] for b in bands], [sensor["std"][b] for b in bands]
            )
            cube = {
                "pixels": torch.from_numpy(raw[None]),
                "time": torch.zeros(1, 4),
                "latlon": torch.zeros(1, 4),
                "gsd": torch.tensor(18.75),
                "waves": torch.tensor([sensor["wavelength"][b] for b in bands]),
            }
            t = model(cube)[0]
            return t[:, 1:].mean(1).numpy()[0], t[:, 0].numpy()[0]
    elif arch == "prithvi":
        path = PACKAGE / "vendor/prithvi_eo_300m/prithvi_mae.py"
        spec = importlib.util.spec_from_file_location("prithvi_pinned", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = read(path.parent / "config.json")["pretrained_cfg"]
        model = module.PrithviMAE(**cfg)
        if not random:
            model.load_state_dict(
                torch.load(weight, weights_only=True, mmap=True, map_location="cpu"), strict=True
            )

        def forward(x):
            t = model.forward_features(
                torch.from_numpy(normalise(x, cfg["mean"], cfg["std"])[None, :, None])
            )[-1]
            return t[:, 1:].mean(1).numpy()[0], t[:, 0].numpy()[0]
    elif name == "teacher":
        import timm
        from torchvision.transforms import v2

        ckpt = torch.load(weight, weights_only=True, mmap=True, map_location="cpu")
        with torch.device("meta"):
            model = timm.create_model(
                ckpt["hyper_parameters"]["teacher"], pretrained=False, num_classes=0
            )
        prefix = "model.teacher."
        model.load_state_dict(
            {k[len(prefix) :]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)},
            strict=True,
            assign=True,
        )

        def forward(x):
            t = v2.Normalize([1552, 1355, 1105], [1888, 1757, 1809])(
                torch.from_numpy(x[[2, 1, 0]].astype(np.float32) * 10000)
            )
            t = v2.Resize((518, 518))(t)[None]
            y = model(t).numpy()[0]
            return y, y
    else:
        import open_clip
        from PIL import Image

        model, _, pre = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=None,
            pretrained_image=False,
            pretrained_text=False,
            load_weights=False,
            force_quick_gelu=name.endswith("quickgelu"),
            device="cpu",
            precision="fp32",
        )
        model.load_state_dict(
            torch.load(weight, weights_only=True, mmap=True, map_location="cpu"), strict=True
        )
        model.eval()
        tokens = open_clip.get_tokenizer("ViT-B-32")(phrases)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            text = model.encode_text(tokens).numpy().copy()

        def forward(x):
            rgb = np.rint(
                255
                * np.clip(np.moveaxis(x[[2, 1, 0]], 0, -1).astype(float) / 0.4, 0, 1) ** (1 / 2.2)
            ).astype(np.uint8)
            y = model.encode_image(pre(Image.fromarray(rgb))[None]).numpy()[0]
            return y, y

    model.eval().requires_grad_(False)
    return model, forward, text


def text_references(work):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from geofm_jlens.semantic_geometry_hierarchy_v1 import unit

    manifest = read(PACKAGE / "data/text-encoders.json")
    classes = read(PACKAGE / "data/text.json")["classes"]
    result = {}
    for entry in manifest:
        local = work / "weights" / entry["name"]
        for asset in entry["files"]:
            url = f"https://huggingface.co/{entry['model']}/resolve/{entry['revision']}/{asset['file']}"
            download(url, local / asset["file"], asset["sha256"])
        qwen = entry["name"] == "qwen"
        tok = AutoTokenizer.from_pretrained(
            local, trust_remote_code=False, padding_side="left" if qwen else "right"
        )
        model = AutoModel.from_pretrained(
            local,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        for mode in ["qwen_plain", "qwen_instructed"] if qwen else ["mpnet_plain"]:
            strings = entry["strings"]
            prefix = (
                "Instruct: Represent the land-cover meaning of this category for semantic similarity.\nQuery: "
                if mode == "qwen_instructed"
                else ""
            )
            batch = tok(
                [prefix + t for t in strings], padding=True, truncation=False, return_tensors="pt"
            )
            with torch.inference_mode():
                h = model(**batch).last_hidden_state
                if qwen:
                    raw = h[:, -1]
                else:
                    mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                    raw = (h * mask).sum(1) / mask.sum(1)
            v = unit(raw.numpy().astype(float))
            result[mode] = np.stack(
                [
                    unit(
                        v[[strings.index(t) for t in [c["definition"], *c["phrases"]]]].mean(0)[
                            None
                        ]
                    )[0]
                    for c in classes
                ]
            )
            result[mode + "_distance"] = 1 - result[mode] @ result[mode].T
        del model
        gc.collect()
    np.savez_compressed(work / "cache/text.npz", **result)


def rebuild(work):
    import torch
    from pyproj import Transformer
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from geofm_jlens.semantic_nuisance_v1 import BANDS, descriptors
    from geofm_jlens.semantic_ridge_v1 import fit_ridge

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    work = Path(work)
    out = work / "cache"
    out.mkdir(exist_ok=True)
    scene = read(PACKAGE / "data/scenes.json")
    bundle = read(PACKAGE / "data/bundle.json")
    rows = scene["patches"]
    labels = read(work / "merged-labels.json")
    phrases = [
        t
        for c in read(PACKAGE / "data/text.json")["classes"]
        for t in [c["definition"], *c["phrases"]]
    ]
    arrays = {}
    names = [
        n
        for n in bundle["feature_names"]
        if not n.endswith("_global") and n not in ["metadata", "appearance"]
    ]
    for name in names:
        path = out / (name + ".npz")
        if not path.exists():
            model, forward, text = encoder(name, work, phrases)
            features = []
            globals_ = []
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                for i, row in enumerate(rows):
                    with np.load(work / "imagery" / (row["patch_id"] + ".npz")) as z:
                        f, g = forward(z["input64"])
                    features.append(f.copy())
                    globals_.append(g.copy())
                    if (i + 1) % 100 == 0:
                        print(name, i + 1, flush=True)
            a = {"features": np.stack(features), "global_": np.stack(globals_)}
            if text is not None:
                a["text"] = text
            np.savez_compressed(path, **a)
            del model, forward
            gc.collect()
        with np.load(path) as z:
            arrays["feature__" + name] = z["features"].astype(float)
            if name + "_global" in bundle["feature_names"]:
                arrays["feature__" + name + "_global"] = z["global_"].astype(float)
            if "text" in z:
                arrays["text__" + name] = z["text"].copy()
    appearance = []
    metadata = []
    quality = []
    oracle = []
    available = []
    for row, label in zip(rows, labels):
        if row["patch_id"] != label["patch_id"]:
            raise ValueError("Label order mismatch")
        with np.load(work / "imagery" / (row["patch_id"] + ".npz")) as z:
            native = z["native20"]
            scl = z["scl"]
            _, a, _ = descriptors(
                dict(zip(BANDS, native)), {b: np.ones((60, 60), bool) for b in BANDS}
            )
            appearance.append(a)
            quality.append([(scl == k).mean() for k in [0, 1, 3, 8, 9, 10, 11]])
        t = row["transform"]
        lon, lat = Transformer.from_crs(row["crs"], "EPSG:4326", always_xy=True).transform(
            t[2] + 600, t[5] - 600
        )
        phi, theta = math.radians(lat), math.radians(lon)
        date = datetime.strptime(row["date"], "%Y-%m-%d")
        phase = (
            2
            * math.pi
            * (date.timetuple().tm_yday - 1)
            / (366 if calendar.isleap(date.year) else 365)
        )
        metadata.append(
            [
                math.cos(phi) * math.cos(theta),
                math.cos(phi) * math.sin(theta),
                math.sin(phi),
                math.sin(phase),
                math.cos(phase),
                date.year,
                int(row["platform"] == "S2A"),
                int(row["platform"] == "S2B"),
            ]
        )
        ok = row["role"] != "fit_single_expert"
        available.append(ok)
        oracle.append(
            [*label["lower"], label["unknown"], label["cloud"]] if ok else [float("nan")] * 11
        )
    arrays.update(
        feature__appearance=np.array(appearance),
        feature__metadata=np.array(metadata),
        nuisance=np.column_stack([appearance, metadata, quality]),
        oracle=np.array(oracle),
        oracle_available=np.array(available),
        membership=np.array([[c in r["positive"] for c in bundle["classes"]] for r in rows]),
    )
    np.savez_compressed(out / "features.npz", **arrays)
    # The nuisance penalties and matching caliper were fixed on development data.
    cal = read(PACKAGE / "data/calibration.json")
    index = {r["patch_id"]: i for i, r in enumerate(rows)}
    ix = [index[p] for p in cal["fit_ids"]]
    x = arrays["nuisance"]
    mean = x[ix].mean(0)
    scale = x[ix].std(0)
    scale[scale == 0] = 1
    values = {"fit_mean": mean, "fit_scale": scale}
    for rec in cal["fits"]:
        name = rec["model"]
        m = fit_ridge(x[ix], arrays["feature__" + name][ix], rec["selected_regularisation"])
        values[name + "-coefficients"] = m.coefficients
        values[name + "-intercept"] = m.y_mean
    np.savez_compressed(out / "calibration.npz", **values)
    text_references(work)
