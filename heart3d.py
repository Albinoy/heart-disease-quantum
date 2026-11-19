"""
CardiaQ - Phase 2: interactive 3D heart viewer (Three.js inside Streamlit).

- Drag to rotate, scroll to zoom, click any part to learn what it does.
- Pass the SHAP region scores from explain.region_scores() and the matching
  parts of the heart pulse red, with a note explaining why.

Usage in app.py:
    from heart3d import render_heart
    render_heart(region_scores(result["contributions"]))   # patient mode
    render_heart()                                          # learn mode

Two kinds of heart model:
1. Built-in (default): generated in code, no files needed.
2. Real anatomical model: put a file called heart.glb next to app.py and it is
   used automatically (or pass model_path="..."). Mesh names are matched to
   heart parts with PART_PATTERNS below, e.g. a mesh named "Left ventricle"
   or "Ventriculus sinister" becomes the left ventricle.

Preview without Streamlit:
    python heart3d.py              -> heart_preview.html (open in Chrome/Safari)
    python heart3d.py --list-meshes heart.glb   -> shows mesh names in a .glb
"""

from __future__ import annotations

import base64
import functools
import json
import os
import re
import struct
import sys

# ---------------------------------------------------------------------------
# Educational information for each clickable part
# ---------------------------------------------------------------------------
PARTS: dict[str, dict[str, str]] = {
    "left_ventricle": {
        "name": "Left ventricle", "short": "Left ventricle",
        "role": "The main pumping chamber. It pushes oxygen-rich blood out through the aorta to the whole body, so it has the thickest muscle wall.",
        "affected": "High blood pressure (the wall thickens), heart attack, heart failure.",
    },
    "right_ventricle": {
        "name": "Right ventricle", "short": "Right ventricle",
        "role": "Pumps oxygen-poor blood into the pulmonary artery and on to the lungs.",
        "affected": "Lung disease, high blood pressure in the lungs, right-sided heart failure.",
    },
    "left_atrium": {
        "name": "Left atrium", "short": "Left atrium",
        "role": "Collects oxygen-rich blood returning from the lungs and passes it to the left ventricle.",
        "affected": "Atrial fibrillation. Clots that form here can travel to the brain and cause a stroke.",
    },
    "right_atrium": {
        "name": "Right atrium", "short": "Right atrium",
        "role": "Collects blood returning from the body. It holds the sinoatrial node, the heart's natural pacemaker.",
        "affected": "Atrial fibrillation and flutter, other rhythm problems.",
    },
    "aorta": {
        "name": "Aorta and arch branches", "short": "Aorta",
        "role": "The body's largest artery. Its arch gives off the arteries to the head and arms, including the carotid arteries that feed the brain.",
        "affected": "Atherosclerosis (plaque), aneurysm, damage from high blood pressure.",
    },
    "pulmonary_artery": {
        "name": "Pulmonary trunk and arteries", "short": "Pulmonary artery",
        "role": "Carries oxygen-poor blood from the right ventricle to both lungs.",
        "affected": "Pulmonary embolism, pulmonary hypertension.",
    },
    "pulmonary_veins": {
        "name": "Pulmonary veins", "short": "Pulmonary veins",
        "role": "Four veins that bring oxygen-rich blood back from the lungs into the left atrium.",
        "affected": "A common starting point for atrial fibrillation.",
    },
    "vena_cava": {
        "name": "Superior and inferior vena cava", "short": "Vena cava",
        "role": "The two large veins that return blood from the upper and lower body to the right atrium.",
        "affected": "Rarely diseased themselves; can be compressed by tumours or blocked by clots.",
    },
    "lad": {
        "name": "Left anterior descending artery (LAD)", "short": "LAD",
        "role": "A coronary artery running down the front groove. It feeds the front of the left ventricle and the wall between the ventricles.",
        "affected": "Plaque build-up. A blockage here causes some of the most serious heart attacks.",
    },
    "rca": {
        "name": "Right coronary artery (RCA)", "short": "RCA",
        "role": "Runs in the groove on the right side. Feeds the right ventricle and, in most people, the bottom of the heart and the electrical nodes.",
        "affected": "Plaque build-up. Blockage can cause heart attack and rhythm problems.",
    },
    "lcx": {
        "name": "Left circumflex artery (LCx)", "short": "LCx",
        "role": "Wraps around the left side of the heart to feed the side and back of the left ventricle.",
        "affected": "Plaque build-up leading to angina or heart attack.",
    },
}

# Which 3D parts light up for each region id used in explain.py
REGION_PARTS: dict[str, list[str]] = {
    "coronary_arteries": ["lad", "rca", "lcx"],
    "left_ventricle": ["left_ventricle"],
    "right_ventricle": ["right_ventricle"],
    "left_atrium": ["left_atrium"],
    "right_atrium": ["right_atrium"],
    "myocardium": ["left_ventricle", "right_ventricle"],
    "systemic_arteries": ["aorta"],
    "pulmonary": ["pulmonary_artery"],
    "whole_heart": [],  # general risk: shown in the legend, nothing blinks
}

REGION_NAMES: dict[str, str] = {
    "coronary_arteries": "Coronary arteries",
    "left_ventricle": "Left ventricle",
    "right_ventricle": "Right ventricle",
    "left_atrium": "Left atrium",
    "right_atrium": "Right atrium",
    "myocardium": "Heart muscle",
    "systemic_arteries": "Aorta and arteries",
    "pulmonary": "Pulmonary artery",
    "whole_heart": "General risk",
}

# Mesh-name patterns for real .glb models (checked in this order, case-insensitive).
# Covers English names, Latin names (Z-Anatomy / BodyParts3D) and CT-segmentation names.
PART_PATTERNS: list[tuple[str, str]] = [
    # Valve leaflets/cusps are shown but never blink (checked first so that e.g.
    # "Left coronary leaflet" is not mistaken for a coronary artery).
    ("_ignore", r"leaflet|cusp|valve|chorda|papillary|node[s]?\b|lymph|scapular|cardiac.?vein|coronary.?sinus|vein.?of.?(left|right).?ventric"),
    ("lad", r"anterior.?descending|interventricular.?(branch|ramus).?anterior|anterior.?interventricular|\blad\b"),
    ("lcx", r"circumflex|circumflexus|\blcx\b"),
    ("rca", r"right.?coronary|coronaria.?dextra|\brca\b"),
    ("lad", r"left.?(main.?)?coronary|coronaria.?sinistra|\blca\b"),
    ("pulmonary_veins", r"pulmonary.?vein|venae?.?pulmonal"),
    ("pulmonary_artery", r"pulmonary.?(trunk|arter)|truncus.?pulmonalis|arteria.?pulmonalis"),
    ("vena_cava", r"vena.?cava|\bcava\b|\b[si]vc\b"),
    ("aorta", r"aort|brachiocephalic|carotid|subclavian"),
    ("left_ventricle", r"left.?ventric|ventriculus.?sinister|myocard|\blv\b"),
    ("right_ventricle", r"right.?ventric|ventriculus.?dexter|\brv\b"),
    ("left_atrium", r"left.?(atri|auricle)|atrium.?sinistrum|auricula.?sinistra"),
    ("right_atrium", r"right.?(atri|auricle)|atrium.?dextrum|auricula.?dextra"),
]


def _normalise(regions) -> list[dict]:
    """Accept a DataFrame from explain.region_scores(), a list of dicts, or None."""
    if regions is None:
        return []
    if hasattr(regions, "to_dict"):
        regions = regions.to_dict("records")
    rows = [r for r in regions if float(r.get("score", 0)) > 0]
    if not rows:
        return []
    top = max(float(r["score"]) for r in rows)
    out = []
    for r in rows:
        rid = r["region"]
        out.append({
            "region": rid,
            "name": REGION_NAMES.get(rid, r.get("region_name", rid)),
            "score": float(r["score"]),
            "intensity": round(0.35 + 0.65 * float(r["score"]) / top, 3),
            "features": r.get("features", ""),
            "parts": REGION_PARTS.get(rid, []),
        })
    return sorted(out, key=lambda r: -r["score"])


@functools.lru_cache(maxsize=4)
def _read_b64(path: str, mtime: float) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _model_b64(model_path):
    if model_path is None and os.path.exists("heart.glb"):
        model_path = "heart.glb"
    if not model_path:
        return None
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"3D model not found: {model_path}")
    return _read_b64(os.path.abspath(model_path), os.path.getmtime(model_path))


def heart_html(regions=None, height: int = 600, model_path: str | None = None) -> str:
    data = {
        "parts": PARTS,
        "regions": _normalise(regions),
        "glb": _model_b64(model_path),
        "patterns": PART_PATTERNS,
    }
    return _TEMPLATE.replace("__DATA__", json.dumps(data))


def render_heart(regions=None, height: int = 600, model_path: str | None = None):
    import streamlit.components.v1 as components

    components.html(heart_html(regions, height, model_path), height=height)


def list_glb_meshes(path: str) -> list[str]:
    """Print mesh/node names in a .glb and which heart part each would map to."""
    with open(path, "rb") as f:
        data = f.read()
    length = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + length])
    names = [n.get("name", "") for n in gltf.get("nodes", [])] + [m.get("name", "") for m in gltf.get("meshes", [])]
    out = []
    for name in dict.fromkeys(n for n in names if n):
        part = next((p for p, pat in PART_PATTERNS if re.search(pat, name, re.I)), "(not matched)")
        part = "(shown, does not blink)" if part == "_ignore" else part
        out.append(f"{name:50s} -> {part}")
    return out


# ---------------------------------------------------------------------------
# HTML + Three.js
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: #1d2433; }
  #wrap { position: relative; width: 100%; height: 100vh; background: radial-gradient(ellipse at 45% 40%, #ffffff 0%, #eceef3 70%, #e2e5ec 100%); border-radius: 14px; overflow: hidden; }
  #info, #legend { position: absolute; background: rgba(255,255,255,0.95); border: 1px solid #e3e6ee; border-radius: 12px; padding: 12px 14px; font-size: 13px; line-height: 1.45; box-shadow: 0 4px 14px rgba(20,30,60,0.06); }
  #info { left: 12px; bottom: 12px; width: 310px; max-width: calc(100% - 24px); }
  #info h3 { margin: 0 0 6px; font-size: 15px; }
  #info .why { margin-top: 8px; padding: 8px; background: #fff1f1; border-left: 3px solid #e03131; border-radius: 6px; }
  #legend { right: 12px; top: 12px; width: 240px; max-height: calc(100% - 24px); overflow: auto; }
  #legend h4 { margin: 0 0 8px; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; color: #5a6275; }
  .reg { margin-bottom: 9px; cursor: pointer; }
  .reg b { display: block; }
  .bar { height: 5px; border-radius: 3px; background: #e03131; margin: 3px 0; }
  .muted { color: #6b7385; font-size: 12px; }
  #tools { position: absolute; left: 12px; top: 12px; display: flex; gap: 6px; }
  #tools button { font: inherit; font-size: 12px; padding: 6px 10px; border-radius: 8px; border: 1px solid #d5d9e3; background: #fff; cursor: pointer; }
  #tools button.on { background: #1d2433; color: #fff; border-color: #1d2433; }
  #msg { position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); color: #6b7385; font-size: 13px; }
  .lbl { font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 10px; background: rgba(29,36,51,0.85); color: #fff; white-space: nowrap; pointer-events: none; }
  .lbl.hot { background: #e03131; }
</style>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/" } }
</script>
</head><body>
<div id="wrap">
  <div id="msg">Loading 3D heart...</div>
  <div id="tools">
    <button id="bRotate" class="on">Auto-rotate</button>
    <button id="bLabels">All labels</button>
    <button id="bReset">Reset view</button>
  </div>
  <div id="legend"></div>
  <div id="info"><h3>Explore the heart</h3><div class="muted">Drag to rotate, scroll to zoom. Click any part to see what it does.</div></div>
</div>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const DATA = __DATA__;
const wrap = document.getElementById("wrap");
const W = () => wrap.clientWidth, H = () => wrap.clientHeight;

// ---------- renderer, scene, camera
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W(), H());
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
wrap.prepend(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(W(), H());
Object.assign(labelRenderer.domElement.style, { position: "absolute", top: "0", left: "0", pointerEvents: "none" });
wrap.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(35, W() / H(), 0.1, 100);
const TARGET = new THREE.Vector3(0, 0.9, 0);
const HOME = new THREE.Vector3(0.3, 1.4, 12);
camera.position.copy(HOME);

const key = new THREE.DirectionalLight(0xfff4ee, 2.2);
key.position.set(4, 7, 7); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0004; key.shadow.normalBias = 0.03;
Object.assign(key.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 1, far: 25 });
scene.add(key);
const rim = new THREE.DirectionalLight(0xffc4bc, 0.9); rim.position.set(-6, 3, -6); scene.add(rim);
scene.add(new THREE.HemisphereLight(0xffffff, 0x705050, 0.35));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.minDistance = 4; controls.maxDistance = 18;
controls.target.copy(TARGET); controls.autoRotate = true; controls.autoRotateSpeed = 1.0;

const heart = new THREE.Group();
scene.add(heart);
const partMeshes = {};                       // part id -> [mesh, ...]
function register(part, mesh) {
  mesh.userData.part = part;
  mesh.castShadow = true; mesh.receiveShadow = true;
  (partMeshes[part] = partMeshes[part] || []).push(mesh);
}
function materials(mesh) { return Array.isArray(mesh.material) ? mesh.material : [mesh.material]; }

// =====================================================================
// A. Built-in heart, generated in code
// =====================================================================
function fibreTexture() {
  // Fine streaks used as a bump map, so the muscle surface is not plastic-smooth.
  const c = document.createElement("canvas"); c.width = c.height = 512;
  const g = c.getContext("2d");
  g.fillStyle = "#808080"; g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 1400; i++) {
    const x = Math.random() * 512, y = Math.random() * 512, len = 20 + Math.random() * 60;
    const v = 100 + Math.random() * 60;
    g.strokeStyle = `rgba(${v},${v},${v},0.35)`; g.lineWidth = 1 + Math.random() * 2;
    g.beginPath(); g.moveTo(x, y); g.lineTo(x + len * 0.8, y + len * 0.6); g.stroke();
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(3, 3);
  return t;
}

function buildBuiltIn() {
  const fibre = fibreTexture();
  const tissue = (color, extra = {}) => new THREE.MeshPhysicalMaterial(Object.assign({
    color, roughness: 0.42, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.32,
    sheen: 0.6, sheenColor: new THREE.Color(0xff9a8a), sheenRoughness: 0.5,
    bumpMap: fibre, bumpScale: 1.2,
  }, extra));
  const vessel = (color) => new THREE.MeshPhysicalMaterial({
    color, roughness: 0.35, clearcoat: 0.8, clearcoatRoughness: 0.2, sheen: 0.3, sheenColor: new THREE.Color(0xffffff) });

  // Organic, rounded "heart body" shape: a sphere stretched into a blunt apex, with soft noise.
  function body(r, apexLen, taper, seed) {
    const g = new THREE.SphereGeometry(r, 128, 96);
    const p = g.attributes.position, v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) {
      v.fromBufferAttribute(p, i);
      const y = v.y / r;
      const s = y < 0 ? 1 + y * taper : 1 - y * y * 0.12;
      v.x *= s; v.z *= s;
      v.y *= y < 0 ? apexLen : 0.82;
      const n = 0.018 * r * Math.sin(v.x * 4.1 + seed) * Math.sin(v.y * 3.3 + seed * 2) * Math.sin(v.z * 5.2 + seed * 3);
      v.addScaledVector(v.clone().normalize(), n);
      p.setXYZ(i, v.x, v.y, v.z);
    }
    g.computeVertexNormals();
    return g;
  }
  function blob(r, seed) { return body(r, 1.0, 0.0, seed); }
  function mesh(part, geo, material, pos, scale, rot) {
    const m = new THREE.Mesh(geo, material);
    if (pos) m.position.set(...pos);
    if (scale) m.scale.set(...scale);
    if (rot) m.rotation.set(...rot);
    heart.add(m); register(part, m);
    return m;
  }
  // Vessel tube with a dark "cut" cap at its open end(s).
  function vesselTube(part, pts, r, material, caps = [1]) {
    const curve = new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(...p)));
    mesh(part, new THREE.TubeGeometry(curve, 96, r, 32, false), material);
    for (const t of caps) {
      const c = new THREE.Mesh(new THREE.CircleGeometry(r * 0.98, 32),
        new THREE.MeshStandardMaterial({ color: 0x4a1418, roughness: 0.8, side: THREE.DoubleSide }));
      const at = curve.getPointAt(t), tan = curve.getTangentAt(t);
      c.position.copy(at); c.lookAt(at.clone().add(tan));
      heart.add(c); register(part, c);
    }
    return curve;
  }

  const LV = tissue(0xa23f3b), RV = tissue(0xab4741);
  const LA = tissue(0xb7574f, { bumpScale: 0.6 }), RA = tissue(0xb35650, { bumpScale: 0.6 });
  const AORTA = vessel(0xc8323a), PA = vessel(0x4b5fae), VC = vessel(0x41539c), PV = vessel(0xb53c48);

  // Ventricles overlap heavily so they read as one muscle mass with a groove between them.
  mesh("left_ventricle",  body(1.12, 1.55, 0.55, 1.0), LV, [0.32, -0.05, -0.12], [1, 1, 0.95], [0, 0, 0.05]);
  mesh("right_ventricle", body(0.95, 1.05, 0.50, 2.0), RV, [-0.42, 0.18, 0.30], [1.02, 1, 0.78], [0, 0.2, -0.08]);

  // Atria with their ear-like auricles.
  mesh("left_atrium",  blob(0.62, 3.0), LA, [0.45, 1.2, -0.72], [1.3, 0.78, 0.9]);
  mesh("left_atrium",  blob(0.3, 3.5), LA, [0.98, 1.3, -0.05], [1.35, 0.55, 0.85], [0.2, -0.6, -0.3]);
  mesh("right_atrium", blob(0.7, 4.0), RA, [-1.02, 0.95, -0.12], [0.85, 1.1, 1.0]);
  mesh("right_atrium", blob(0.32, 4.5), RA, [-0.62, 1.38, 0.42], [1.3, 0.55, 0.85], [0.3, 0.6, 0.25]);

  // Aorta: ascending, arch, descending, plus the three arch branches.
  vesselTube("aorta", [[0.0,0.85,0.02],[-0.12,1.6,0.15],[-0.08,2.35,0.06],[0.25,2.85,-0.32],[0.78,2.82,-0.78],[1.02,2.3,-1.05],[1.06,1.3,-1.12],[1.06,0.0,-1.12]], 0.31, AORTA);
  vesselTube("aorta", [[0.02,2.72,-0.16],[-0.12,3.15,-0.12],[-0.3,3.7,-0.08]], 0.12, AORTA);
  vesselTube("aorta", [[0.36,2.92,-0.42],[0.38,3.35,-0.42],[0.42,3.85,-0.42]], 0.09, AORTA);
  vesselTube("aorta", [[0.66,2.9,-0.66],[0.82,3.3,-0.7],[1.05,3.65,-0.72]], 0.1, AORTA);

  // Pulmonary trunk from the right ventricle, splitting into left and right pulmonary arteries.
  vesselTube("pulmonary_artery", [[-0.28,0.85,0.58],[-0.18,1.55,0.66],[0.08,2.1,0.42],[0.32,2.3,0.05]], 0.28, PA, []);
  vesselTube("pulmonary_artery", [[0.3,2.3,0.06],[0.75,2.38,-0.25],[1.35,2.32,-0.42]], 0.18, PA);
  vesselTube("pulmonary_artery", [[0.3,2.3,0.06],[-0.15,2.28,-0.42],[-0.8,2.24,-0.55],[-1.45,2.2,-0.55]], 0.17, PA);

  // Venae cavae into the right atrium, pulmonary veins into the left atrium.
  vesselTube("vena_cava", [[-1.02,1.35,-0.16],[-1.04,2.3,-0.2],[-1.04,3.25,-0.24]], 0.21, VC);
  vesselTube("vena_cava", [[-1.0,0.55,-0.3],[-1.0,0.05,-0.38],[-1.0,-0.45,-0.42]], 0.23, VC);
  for (const [a, b] of [[[0.05,1.38,-0.95],[-0.42,1.48,-1.35]], [[0.1,1.02,-0.98],[-0.38,0.98,-1.4]],
                        [[0.92,1.38,-0.95],[1.38,1.5,-1.35]], [[0.88,1.02,-0.98],[1.36,0.98,-1.4]]]) {
    const mid = a.map((v, i) => (v + b[i]) / 2);
    vesselTube("pulmonary_veins", [a, mid, b], 0.1, PV);
  }

  // ---- Coronary arteries: found by ray-casting onto the muscle, so they sit
  //      exactly in the grooves between chambers, whatever the shapes are.
  heart.updateMatrixWorld(true);
  const chambers = ["left_ventricle", "right_ventricle", "left_atrium", "right_atrium"].flatMap(k => partMeshes[k]);
  const ray = new THREE.Raycaster();
  const AXIS = new THREE.Vector2(0.05, 0.05);
  function hitAt(azDeg, y) {
    const a = THREE.MathUtils.degToRad(azDeg), dir = new THREE.Vector3(Math.sin(a), 0, Math.cos(a));
    const origin = new THREE.Vector3(AXIS.x, y, AXIS.y).addScaledVector(dir, 8);
    ray.set(origin, dir.clone().negate());
    const h = ray.intersectObjects(chambers, false)[0];
    return h ? { part: h.object.userData.part, point: h.point.clone().addScaledVector(dir, 0.04) } : null;
  }
  // Scan sideways at height y for where the surface switches from chamber A to chamber B.
  function grooveAcross(y, az0, az1, A, B) {
    let prev = null;
    for (let az = az0; az0 < az1 ? az <= az1 : az >= az1; az += az0 < az1 ? 1.5 : -1.5) {
      const h = hitAt(az, y);
      if (prev && h && prev.part === A && h.part === B) return { az, point: h.point };
      if (h) prev = h;
    }
    return null;
  }
  // Scan downwards at angle az for where an atrium meets a ventricle (the AV groove).
  function grooveDown(az) {
    let prev = null;
    for (let y = 2.0; y >= -0.6; y -= 0.03) {
      const h = hitAt(az, y);
      if (prev && h && prev.part.endsWith("atrium") && h.part.endsWith("ventricle")) return h.point;
      if (h) prev = h;
    }
    return null;
  }
  const corMats = {};   // one material per artery, so each can blink on its own
  const corMat = p => corMats[p] = corMats[p] || new THREE.MeshPhysicalMaterial({ color: 0xe11d2a, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.15 });
  function artery(part, pts, r) {
    pts = pts.filter(Boolean);
    if (pts.length < 3) return;
    const m = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 160, r, 12, false), corMat(part));
    heart.add(m); register(part, m);
  }
  const smooth = (pts, k = 2) => pts.filter((_, i) => i % k === 0 || i === pts.length - 1);

  // LAD: the front groove between right and left ventricle, then on to the apex.
  const lad = []; let lastAz = 20;
  for (let y = 1.0; y >= -1.0; y -= 0.1) {
    const g = grooveAcross(y, -40, 110, "right_ventricle", "left_ventricle");
    if (g) { lad.push(g.point); lastAz = g.az; }
  }
  for (let y = (lad.length ? lad[lad.length - 1].y : 0) - 0.15, k = 1; y >= -1.6 && k < 8; y -= 0.15, k++) {
    const h = hitAt(lastAz + k * 4, y); if (h) lad.push(h.point);
  }
  artery("lad", smooth(lad), 0.055);
  // Diagonal branches off the LAD onto the left ventricle.
  for (const frac of [0.3, 0.55]) {
    const s = lad[Math.floor(lad.length * frac)]; if (!s) continue;
    const az0 = THREE.MathUtils.radToDeg(Math.atan2(s.x - AXIS.x, s.z - AXIS.y));
    const br = [s]; for (let k = 1; k <= 5; k++) { const h = hitAt(az0 + k * 9, s.y - k * 0.12); if (h) br.push(h.point); }
    artery("lad", br, 0.032);
  }
  // LCx: AV groove round the left side; RCA: AV groove round the right side and back.
  const lcx = []; for (let az = 35; az <= 170; az += 6) lcx.push(grooveDown(az));
  artery("lcx", smooth(lcx.filter(Boolean)), 0.05);
  const rca = []; for (let az = -5; az >= -200; az -= 6) rca.push(grooveDown(az));
  const rcaPts = smooth(rca.filter(Boolean));
  // Posterior descending branch at the back of the heart.
  const back = []; for (let y = 0.6; y >= -1.0; y -= 0.1) { const g = grooveAcross(y, 275, 150, "right_ventricle", "left_ventricle"); if (g) back.push(g.point); }
  artery("rca", rcaPts.concat(smooth(back)), 0.05);
  // Marginal branch of the RCA down the right ventricle.
  const mStart = rcaPts[Math.floor(rcaPts.length * 0.35)];
  if (mStart) {
    const az0 = THREE.MathUtils.radToDeg(Math.atan2(mStart.x - AXIS.x, mStart.z - AXIS.y));
    const br = [mStart]; for (let k = 1; k <= 6; k++) { const h = hitAt(az0 + k * 2, mStart.y - k * 0.15); if (h) br.push(h.point); }
    artery("rca", br, 0.032);
  }

  // Real hearts tilt: apex points down, forwards and to the patient's left.
  heart.rotation.set(0.18, -0.3, 0.42);
}

// =====================================================================
// B. Real anatomical model from a .glb file
// =====================================================================
async function buildFromGLB(b64) {
  const { GLTFLoader } = await import("three/addons/loaders/GLTFLoader.js");
  const { DRACOLoader } = await import("three/addons/loaders/DRACOLoader.js");
  const loader = new GLTFLoader();
  const draco = new DRACOLoader(); draco.setDecoderPath("https://www.gstatic.com/draco/versioned/decoders/1.5.6/");
  loader.setDRACOLoader(draco);
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const gltf = await new Promise((ok, fail) => loader.parse(bytes.buffer, "", ok, fail));
  const model = gltf.scene;
  const patterns = DATA.patterns.map(([p, re]) => [p, new RegExp(re, "i")]);
  model.traverse(o => {
    if (!o.isMesh) return;
    let name = "", n = o; while (n) { name += " " + (n.name || ""); n = n.parent; }
    const match = patterns.find(([, re]) => re.test(name));
    o.material = materials(o).map(m => {
      let c = m.clone();
      if (!("emissive" in c)) c = new THREE.MeshStandardMaterial({ color: c.color || 0xb04545, map: c.map || null });
      return c;
    });
    if (o.material.length === 1) o.material = o.material[0];
    if (match && match[0] !== "_ignore") register(match[0], o); else { o.castShadow = o.receiveShadow = true; }
  });
  // Centre and scale the model to the same size as the built-in heart.
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3()).length(), c = box.getCenter(new THREE.Vector3());
  const s = 5.5 / size;
  model.scale.setScalar(s);
  model.position.copy(c.multiplyScalar(-s)).add(TARGET);
  heart.add(model);
  if (!Object.keys(partMeshes).length) console.warn("No mesh names matched PART_PATTERNS; run `python heart3d.py --list-meshes heart.glb`.");
}

// ---------- build
const msg = document.getElementById("msg");
try {
  if (DATA.glb) await buildFromGLB(DATA.glb); else buildBuiltIn();
  msg.remove();
} catch (e) {
  console.error(e);
  msg.textContent = "Could not load the 3D model (" + e.message + "). Showing the built-in heart.";
  if (DATA.glb) { heart.clear(); for (const k in partMeshes) delete partMeshes[k]; buildBuiltIn(); }
}
heart.updateMatrixWorld(true);
for (const ms of Object.values(partMeshes)) for (const m of ms) for (const mat of materials(m)) mat.userData.base = mat.color.clone();

// ---------- labels (one per part, placed just outside the part)
const centre = new THREE.Box3().setFromObject(heart).getCenter(new THREE.Vector3());
const labels = {};
for (const [part, ms] of Object.entries(partMeshes)) {
  const box = new THREE.Box3(); ms.forEach(m => box.expandByObject(m));
  const c = box.getCenter(new THREE.Vector3());
  const out = c.clone().sub(centre).setY(0); if (out.lengthSq() < 1e-4) out.set(0, 0, 1);
  const pos = c.addScaledVector(out.normalize(), 0.45);
  const div = document.createElement("div"); div.className = "lbl"; div.textContent = DATA.parts[part].short;
  const obj = new CSS2DObject(div); obj.position.copy(heart.worldToLocal(pos)); obj.visible = false;
  heart.add(obj); labels[part] = obj;
}

// ---------- highlights from the SHAP regions
const hot = {};
for (const r of DATA.regions) for (const p of r.parts) {
  if (!partMeshes[p]) continue;
  hot[p] = hot[p] || { intensity: 0, reasons: [] };
  hot[p].intensity = Math.max(hot[p].intensity, r.intensity);
  hot[p].reasons.push(`${r.name}: ${r.features}`);
}
const anyHot = Object.keys(hot).length > 0;
const GREY = new THREE.Color(0x9ca1ab);
for (const [part, ms] of Object.entries(partMeshes)) {
  if (hot[part]) { labels[part].visible = true; labels[part].element.classList.add("hot"); }
  else if (anyHot) for (const m of ms) for (const mat of materials(m)) mat.color.copy(mat.userData.base).lerp(GREY, 0.45);
}

// ---------- legend
const legend = document.getElementById("legend");
if (DATA.regions.length) {
  legend.innerHTML = "<h4>Linked to this result</h4>" + DATA.regions.map(r => `
    <div class="reg" data-parts="${r.parts.join(",")}"><b>${r.name}</b>
      <div class="bar" style="width:${Math.round(r.intensity * 100)}%; opacity:${r.parts.length ? 1 : 0.35}"></div>
      <div class="muted">${r.features}${r.parts.length ? "" : " (affects the whole heart, nothing blinks)"}</div></div>`).join("")
    + '<div class="muted">Blink strength shows how much each region pushed the risk up.</div>';
  legend.querySelectorAll(".reg").forEach(el => el.onclick = () => {
    const p = el.dataset.parts.split(",").filter(x => partMeshes[x])[0]; if (p) showInfo(p);
  });
} else {
  legend.innerHTML = "<h4>Learn mode</h4><div class='muted'>No patient result yet. Click any part of the heart to learn about it.</div>";
}

// ---------- info panel + clicking
const info = document.getElementById("info");
let selected = null;
function showInfo(part) {
  const d = DATA.parts[part]; selected = part;
  info.innerHTML = `<h3>${d.name}</h3><div>${d.role}</div>
    <div class="muted" style="margin-top:6px"><b>Can be affected by:</b> ${d.affected}</div>` +
    (hot[part] ? `<div class="why"><b>Lit up for this patient because</b><br>${hot[part].reasons.join("<br>")}</div>` : "");
}
const bRotate = document.getElementById("bRotate"), bLabels = document.getElementById("bLabels");
const picker = new THREE.Raycaster(), pointer = new THREE.Vector2(); let downAt = null;
renderer.domElement.addEventListener("pointerdown", e => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("pointerup", e => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
  picker.setFromCamera(pointer, camera);
  const hit = picker.intersectObjects(Object.values(partMeshes).flat(), false)[0];
  if (hit) { showInfo(hit.object.userData.part); controls.autoRotate = false; bRotate.classList.remove("on"); }
});

// ---------- buttons
bRotate.onclick = () => { controls.autoRotate = !controls.autoRotate; bRotate.classList.toggle("on"); };
let allLabels = false;
bLabels.onclick = () => {
  allLabels = !allLabels; bLabels.classList.toggle("on");
  for (const [p, l] of Object.entries(labels)) l.visible = allLabels || !!hot[p];
};
document.getElementById("bReset").onclick = () => { camera.position.copy(HOME); controls.target.copy(TARGET); };

// ---------- animation: pulsing glow on hot parts, gentle heartbeat
const RED = new THREE.Color(0xff1a1a), SEL = new THREE.Color(0x3a7bff);
const clock = new THREE.Clock();
const baseScale = heart.scale.clone();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();
  const blink = 0.5 + 0.5 * Math.sin(t * 5);
  const beat = 1 + 0.02 * Math.max(0, Math.sin(t * 7.5)) ** 8;
  heart.scale.copy(baseScale).multiplyScalar(beat);
  for (const [part, ms] of Object.entries(partMeshes)) for (const m of ms) for (const mat of materials(m)) {
    if (!mat.emissive) continue;
    if (hot[part]) { mat.emissive.copy(RED); mat.emissiveIntensity = hot[part].intensity * (0.1 + 0.9 * blink); }
    else if (part === selected) { mat.emissive.copy(SEL); mat.emissiveIntensity = 0.3; }
    else mat.emissiveIntensity = 0;
  }
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}
animate();

window.addEventListener("resize", () => {
  camera.aspect = W() / H(); camera.updateProjectionMatrix();
  renderer.setSize(W(), H()); labelRenderer.setSize(W(), H());
});
</script>
</body></html>
"""


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--list-meshes":
        print("\n".join(list_glb_meshes(sys.argv[2])))
        sys.exit()
    demo = [
        {"region": "coronary_arteries", "score": 0.18, "features": "Narrowed major vessels, Chest pain type"},
        {"region": "myocardium", "score": 0.09, "features": "ST depression, Chest pain during exercise"},
        {"region": "whole_heart", "score": 0.03, "features": "Age"},
    ]
    with open("heart_preview.html", "w") as f:
        f.write(heart_html(demo))
    with open("heart_learn_mode.html", "w") as f:
        f.write(heart_html(None))
    print("Wrote heart_preview.html and heart_learn_mode.html - open them in your browser.")
