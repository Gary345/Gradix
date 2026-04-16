from pathlib import Path
import json, textwrap

root = Path(r'e:\Personal\Diplomado\Clases\MODULO 5\Proyecto Final\Gradix')
out = root / 'notebooks' / '01_dataset_pipeline_fuerte_gradix.ipynb'

def md(text):
    return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(text).strip('\n').splitlines(keepends=True)}

def code(text):
    return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':textwrap.dedent(text).strip('\n').splitlines(keepends=True)}

cells = []

cells += [
md('''
# Introduccion

Este notebook crea una version **autosuficiente y fuerte** del pipeline de Gradix para generacion de dataset. Conserva la logica conceptual principal del proyecto real:

**carga -> preprocesado -> deteccion -> warp -> validacion post-warp -> features -> scoring -> estado de analisis -> aplanado -> dataset**.

La app es la interfaz final para usuario. Este notebook es la version reproducible de la construccion del dataset. Ambos comparten la misma logica tecnica, pero aqui se elimina la capa UI y tambien se excluyen Streamlit, OCR, TCGdex, OpenAI y servicios externos.
'''),
md('''
# Fuente De Datos

Se asume una carpeta de imagenes organizada por clase, por ejemplo:

```text
imagenes/
  damaged/
  undamaged/
```

- `label_condition` se deriva de la carpeta padre.
- `target_damaged` convierte esa etiqueta a binario operacional.
- `analysis_status` y `usable_for_condition_model` se calculan tecnicamente a partir del pipeline.
'''),
md('''
# Limitaciones

- Las etiquetas provienen de carpetas y no de grading profesional.
- Hay sesgo posible por fondo, iluminacion y blur.
- Puede haber multiples fotos de la misma carta.
- El target es operacional.
- El notebook porta el nucleo del pipeline, no todo el repo.
'''),
code('''
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

plt.rcParams['figure.dpi'] = 130
INPUT_DIR = Path('data/raw')
OUTPUT_CSV = Path('data/processed/dataset_pipeline_fuerte_gradix.csv')
PACKAGE_ROOT = Path.cwd() / '_gradix_local'
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
print(INPUT_DIR, OUTPUT_CSV, PACKAGE_ROOT)
'''),
md('''
# Creacion De Modulos Locales Del Pipeline

El notebook genera un paquete local `_gradix_local/` con seis modulos: `utils`, `vision`, `features`, `scoring`, `pipeline` y `dataset_builder`.
'''),
code('''
PACKAGE_ROOT.mkdir(parents=True, exist_ok=True)
(PACKAGE_ROOT / '__init__.py').write_text('', encoding='utf-8')
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))
'''),
code(r'''
%%writefile _gradix_local/utils.py
from __future__ import annotations
import cv2, numpy as np

def clamp(v, a, b): return max(a, min(v, b))
def safe_float(v):
    try:
        if v is None: return 0.0
        if isinstance(v, (np.floating, np.integer)): v = float(v)
        if np.isnan(v) or np.isinf(v): return 0.0
        return float(v)
    except Exception: return 0.0

def normalize_to_0_1(v, a, b):
    if b <= a: return 0.0
    return clamp((v-a)/(b-a), 0.0, 1.0)

def ensure_bgr_uint8(image):
    if image is None: raise ValueError('image is None')
    if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3: raise ValueError('shape invalid')
    if image.dtype != np.uint8: image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4,2)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def polygon_area(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1,2)
    x = points[:,0]; y = points[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))

def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        nk = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict): items.update(flatten_dict(v, nk, sep=sep))
        elif isinstance(v, (int,float,str,bool,np.integer,np.floating,np.bool_)) or v is None: items[nk] = v
    return items

def infer_label_from_path(path, raw_root):
    try:
        rel = path.relative_to(raw_root)
        if len(rel.parts) >= 2: return rel.parts[0].lower().strip()
    except Exception: pass
    return 'unknown'

def normalize_label_condition(label):
    label = (label or '').lower().strip()
    if label in {'damaged','damage','danada','dañada','rota','rotas'}: return 'damaged'
    if label in {'undamaged','clean','sana','ok','sin_danio','sin_daño'}: return 'undamaged'
    return 'unknown'

def target_from_label(label):
    if label == 'damaged': return 1
    if label == 'undamaged': return 0
    return None
'''),
]
cells += [
code(r'''
%%writefile _gradix_local/vision.py
from __future__ import annotations
import cv2, numpy as np
from _gradix_local.utils import ensure_bgr_uint8, order_points
TARGET_ASPECT = 0.715

def resize_for_detection(image, target_long_side=1400):
    image = ensure_bgr_uint8(image); h,w = image.shape[:2]; long_side = max(h,w)
    if long_side <= target_long_side: return image.copy(), 1.0
    s = target_long_side / float(long_side)
    return cv2.resize(image, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA), s

def build_multilayer_views(image):
    image = ensure_bgr_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    med = float(np.median(gray)); low = int(max(0,0.66*med)); high = int(min(255,1.33*med))
    canny = cv2.Canny(gray, low, high)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    _, grad_bin = cv2.threshold(grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,6)
    _, otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    maps = {'gray': gray,'canny_closed': cv2.morphologyEx(canny, cv2.MORPH_CLOSE, k, iterations=2),'adaptive_closed': cv2.morphologyEx(adap, cv2.MORPH_CLOSE, k, iterations=2),'otsu_closed': cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k, iterations=2),'grad_closed': cv2.morphologyEx(grad_bin, cv2.MORPH_CLOSE, k, iterations=2)}
    structural = None
    for key in ('canny_closed','adaptive_closed','otsu_closed','grad_closed'):
        structural = maps[key].copy() if structural is None else cv2.bitwise_or(structural, maps[key])
    structural = cv2.morphologyEx(structural, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), iterations=1)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[:,:,1] = cv2.max(overlay[:,:,1], structural)
    overlay[:,:,2] = cv2.max(overlay[:,:,2], maps['canny_closed'])
    maps['structural_mask'] = structural; maps['edge_map'] = maps['canny_closed']; maps['multilayer_overlay'] = overlay
    return maps

def _aspect_ratio(quad):
    quad = order_points(quad)
    top = np.linalg.norm(quad[1]-quad[0]); right = np.linalg.norm(quad[2]-quad[1]); bottom = np.linalg.norm(quad[2]-quad[3]); left = np.linalg.norm(quad[3]-quad[0])
    return float(max(1.0,(top+bottom)/2.0) / max(1.0,(right+left)/2.0))

def _bbox(quad, shape):
    h,w = shape[:2]; quad = order_points(quad)
    x1 = int(np.clip(np.floor(quad[:,0].min()), 0, max(0,w-1))); y1 = int(np.clip(np.floor(quad[:,1].min()),0,max(0,h-1)))
    x2 = int(np.clip(np.ceil(quad[:,0].max()), x1+1, max(x1+1,w))); y2 = int(np.clip(np.ceil(quad[:,1].max()), y1+1, max(y1+1,h)))
    return x1,y1,x2,y2

def _score_candidate(quad, contour, shape, edge_map):
    h,w = shape[:2]; area = float(abs(cv2.contourArea(contour))); area_ratio = area / float(max(1,h*w)); bbox = _bbox(quad, shape); bbox_area = float(max(1,(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))); rectangularity = min(1.0, area/bbox_area)
    aspect = _aspect_ratio(quad); aspect_quality = max(0.0, 1.0 - abs(aspect - TARGET_ASPECT)/0.18); center = np.mean(order_points(quad), axis=0); img_center = np.array([w/2.0,h/2.0], dtype=np.float32); center_score = max(0.0, 1.0 - np.linalg.norm(center-img_center)/max(1e-6,np.linalg.norm(img_center)))
    margins = np.array([quad[:,0].min(), w-quad[:,0].max(), quad[:,1].min(), h-quad[:,1].max()], dtype=np.float32) / float(max(1,min(h,w))); margin_score = min(1.0, float(np.sort(margins)[1]) / 0.03)
    thickness = max(2, int(round(min(h,w)*0.012))); mask = np.zeros((h,w), dtype=np.uint8); cv2.polylines(mask,[order_points(quad).astype(np.int32)],True,255,thickness); active = edge_map[mask>0]; support = min(1.0, float(np.mean(active>0))/0.30) if active.size else 0.0
    score = float(0.33*min(1.0, area_ratio/0.60)+0.22*aspect_quality+0.18*rectangularity+0.12*support+0.10*center_score+0.05*margin_score)
    return {'score':score,'area_ratio':float(area_ratio),'rectangularity':float(rectangularity),'aspect_ratio':float(aspect),'center_score':float(center_score),'edge_support':float(support)}

def _search_best(mask, edge_map, shape, source):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); best=None; best_score=-1.0
    for contour in contours:
        area = float(cv2.contourArea(contour)); area_ratio = area / float(max(1,shape[0]*shape[1]))
        if area_ratio < 0.08 or area_ratio > 0.98: continue
        peri = cv2.arcLength(contour, True); approx = cv2.approxPolyDP(contour, 0.02*peri, True); used_fallback = len(approx) != 4
        quad = order_points(approx.reshape(4,2).astype(np.float32)) if len(approx)==4 else order_points(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32))
        metrics = _score_candidate(quad, contour, shape, edge_map)
        if metrics['score'] > best_score: best_score = metrics['score']; best={'quad':quad,'contour':contour,'used_fallback':used_fallback,'source':source,'metrics':metrics}
    return best

def detect_card_contour(image):
    image = ensure_bgr_uint8(image); resized, scale = resize_for_detection(image); views = build_multilayer_views(resized); best=None; best_score=-1.0
    for key in ('structural_mask','canny_closed','adaptive_closed','otsu_closed','grad_closed'):
        cand = _search_best(views[key], views['edge_map'], resized.shape[:2], key)
        if cand is not None and cand['metrics']['score'] > best_score: best, best_score = cand, cand['metrics']['score']
    debug = {k:views[k] for k in ('gray','canny_closed','adaptive_closed','otsu_closed','structural_mask','multilayer_overlay')}
    if best is None:
        debug['detected_contour_overlay'] = resized.copy()
        return {'success':False,'contour':None,'corners':None,'used_fallback':True,'debug_images':debug,'metrics':{'scale':float(scale),'best_score':0.0,'detection_confidence':0.0,'weak_detection':True,'candidate_source':'none'}}
    bbox = _bbox(best['quad'], resized.shape[:2]); mx = int((bbox[2]-bbox[0])*0.18); my = int((bbox[3]-bbox[1])*0.18); x1,y1,x2,y2 = max(0,bbox[0]-mx), max(0,bbox[1]-my), min(resized.shape[1],bbox[2]+mx), min(resized.shape[0],bbox[3]+my)
    roi = resized[y1:y2, x1:x2]
    if roi.size:
        roi_views = build_multilayer_views(roi); roi_best=None; roi_score=-1.0
        for key in ('structural_mask','canny_closed','adaptive_closed','otsu_closed','grad_closed'):
            cand = _search_best(roi_views[key], roi_views['edge_map'], roi.shape[:2], 'roi_'+key)
            if cand is not None and cand['metrics']['score'] > roi_score: roi_best, roi_score = cand, cand['metrics']['score']
        if roi_best is not None and roi_best['metrics']['score'] > best['metrics']['score'] + 0.03: roi_best['quad'] = order_points(roi_best['quad'] + np.array([x1,y1], dtype=np.float32)); best = roi_best
        ov = resized.copy(); cv2.rectangle(ov,(x1,y1),(x2,y2),(255,180,0),2); debug['roi_second_pass_overlay'] = ov
    quad_resized = order_points(best['quad']); draw = resized.copy(); cv2.polylines(draw,[quad_resized.astype(np.int32)],True,(0,255,0),3); debug['detected_contour_overlay'] = draw
    corners_original = (quad_resized / scale if scale != 1.0 else quad_resized.copy()).astype(np.float32); contour_original = corners_original.astype(np.int32).reshape(-1,1,2)
    conf = float(np.clip(0.50*best['metrics']['score']+0.20*best['metrics']['rectangularity']+0.20*best['metrics']['edge_support']+0.10*best['metrics']['center_score'],0.0,1.0)); weak = bool(best['used_fallback'] or conf < 0.55 or best['metrics']['score'] < 0.60)
    metrics = dict(best['metrics']); metrics.update({'scale':float(scale),'best_score':float(best['metrics']['score']),'detection_confidence':conf,'weak_detection':weak,'candidate_source':best['source']})
    return {'success':True,'contour':contour_original,'corners':corners_original,'used_fallback':bool(best['used_fallback']),'debug_images':debug,'metrics':metrics}

def warp_card_perspective(image, corners, target_aspect_ratio=0.714, expand_ratio=0.012, min_output_height=700, max_output_height=1200):
    bgr = ensure_bgr_uint8(image); ordered = order_points(corners); center = np.mean(ordered, axis=0, keepdims=True); expanded = (center + (ordered-center)*(1.0+expand_ratio)).astype(np.float32)
    top = np.linalg.norm(expanded[1]-expanded[0]); right = np.linalg.norm(expanded[2]-expanded[1]); bottom = np.linalg.norm(expanded[2]-expanded[3]); left = np.linalg.norm(expanded[3]-expanded[0]); raw_w = max(1.0, (top+bottom)/2.0); raw_h = max(1.0, (left+right)/2.0)
    out_h = max(min_output_height, min(max_output_height, int(round(raw_h)))); out_w = max(300, int(round(out_h*target_aspect_ratio))); out_h = max(420,out_h)
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32); M = cv2.getPerspectiveTransform(expanded, dst); warped = cv2.warpPerspective(bgr, M, (out_w,out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return {'success':True,'warped_image':warped,'ordered_corners':expanded,'transform_matrix':M.astype(np.float32),'output_size':(int(out_w),int(out_h)),'metrics':{'expand_ratio':float(expand_ratio),'raw_width':float(raw_w),'raw_height':float(raw_h),'raw_aspect_ratio':float(raw_w/raw_h),'output_width':int(out_w),'output_height':int(out_h),'target_aspect_ratio':float(target_aspect_ratio)}}

def validate_rectified_card(warped_card_bgr):
    bgr = ensure_bgr_uint8(warped_card_bgr); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); h,w = gray.shape[:2]; aspect = float(w/max(1,h)); dev = abs(aspect - 0.714); aspect_score = max(0.0, 1.0-dev/0.18)
    bd = max(5,int(round(min(h,w)*0.03))); idp = max(bd+2,int(round(min(h,w)*0.07))); diffs = [abs(float(np.mean(gray[:bd,:]))-float(np.mean(gray[bd:idp,:]))), abs(float(np.mean(gray[h-bd:,:]))-float(np.mean(gray[h-idp:h-bd,:]))), abs(float(np.mean(gray[:,:bd]))-float(np.mean(gray[:,bd:idp]))), abs(float(np.mean(gray[:,w-bd:]))-float(np.mean(gray[:,w-idp:w-bd])))]
    border_score = float(min(1.0, np.mean(diffs)/45.0)); crop_risk = float(max(0.0, 1.0-min(1.0,np.mean(diffs)/35.0))); margin_consistency = float(max(0.0, 1.0-np.std(diffs)/40.0)); continuity = float(min(1.0, np.std(gray)/80.0)); postwarp_score = float(np.clip(0.24*aspect_score+0.26*border_score+0.18*continuity+0.16*margin_consistency+0.16*(1.0-crop_risk),0.0,1.0))
    valid = bool(postwarp_score >= 0.56 and aspect_score >= 0.35 and border_score >= 0.30); retry = bool((not valid) or postwarp_score < 0.64 or crop_risk > 0.55)
    return {'postwarp_valid':valid,'postwarp_score':postwarp_score,'retry_recommended':retry,'rectified_aspect_ratio':aspect,'rectified_aspect_ratio_deviation':dev,'rectified_aspect_ratio_score':aspect_score,'outer_border_score':border_score,'margin_consistency_score':margin_consistency,'crop_risk_score':crop_risk,'outer_border_continuity_score':continuity}
'''),
code(r'''
%%writefile _gradix_local/features.py
from __future__ import annotations
import cv2, numpy as np
from _gradix_local.utils import ensure_bgr_uint8, clamp, normalize_to_0_1, polygon_area

def extract_visual_features(image_bgr):
    image_bgr = ensure_bgr_uint8(image_bgr); gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY); h,w = image_bgr.shape[:2]
    return {'width_px':int(w),'height_px':int(h),'aspect_ratio':float(w/max(1.0,h)),'area_px':int(w*h),'blur_score':float(cv2.Laplacian(gray, cv2.CV_64F).var()),'brightness_score':float(np.mean(gray)),'contrast_score':float(np.std(gray))}

def extract_geometry_features(contour, image_shape, warped_aspect_ratio, used_fallback):
    coverage = 0.0 if contour is None else float(polygon_area(contour.reshape(-1,2)) / max(1.0, image_shape[0]*image_shape[1])); aspect_quality = clamp(1.0 - abs(warped_aspect_ratio-0.715)/0.10, 0.0, 1.0)
    return {'coverage_ratio':round(coverage,3),'aspect_ratio_quality':round(aspect_quality,3),'used_fallback':bool(used_fallback)}

def _find_content_bbox(image_bgr):
    gray = cv2.cvtColor(ensure_bgr_uint8(image_bgr), cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray,(5,5),0); gx = cv2.Sobel(gray, cv2.CV_32F,1,0,ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F,0,1,ksize=3)
    col = cv2.GaussianBlur(np.abs(gx).mean(axis=0).reshape(1,-1),(1,31),0).ravel(); row = cv2.GaussianBlur(np.abs(gy).mean(axis=1).reshape(-1,1),(31,1),0).ravel(); h,w = gray.shape[:2]
    xl = col[int(w*0.05):int(w*0.35)]; xr = col[int(w*0.65):int(w*0.95)]; yt = row[int(h*0.05):int(h*0.25)]; yb = row[int(h*0.75):int(h*0.95)]
    x_min = int(w*0.05)+int(np.argmax(xl)) if xl.size else int(w*0.12); x_max = int(w*0.65)+int(np.argmax(xr)) if xr.size else int(w*0.88); y_min = int(h*0.05)+int(np.argmax(yt)) if yt.size else int(h*0.10); y_max = int(h*0.75)+int(np.argmax(yb)) if yb.size else int(h*0.90)
    if x_max <= x_min or y_max <= y_min: x_min,y_min,x_max,y_max = int(w*0.12),int(h*0.10),int(w*0.88),int(h*0.90)
    return {'x_min':int(x_min),'y_min':int(y_min),'x_max':int(x_max),'y_max':int(y_max)}

def extract_centering_features(image_bgr):
    h,w = image_bgr.shape[:2]; bbox = _find_content_bbox(image_bgr); l = float(bbox['x_min']); r = float(w-bbox['x_max']); t = float(bbox['y_min']); b = float(h-bbox['y_max']); bal = lambda a,b: 0.0 if (a+b)<=0 else max(0.0, 1.0-abs(a-b)/(a+b)); hb, vb = bal(l,r), bal(t,b)
    return {'content_bbox':bbox,'left_margin':round(l,2),'right_margin':round(r,2),'top_margin':round(t,2),'bottom_margin':round(b,2),'horizontal_balance':round(hb,3),'vertical_balance':round(vb,3),'overall_centering':round((hb+vb)/2.0,3)}

def draw_content_bbox(image_bgr, bbox):
    out = ensure_bgr_uint8(image_bgr).copy(); cv2.rectangle(out,(bbox['x_min'],bbox['y_min']),(bbox['x_max'],bbox['y_max']),(255,255,255),3); return out
'''),
]
cells += [
code(r'''
%%writefile _gradix_local/scoring.py
from __future__ import annotations
from _gradix_local.utils import clamp

def compute_capture_quality_score(features):
    blur = clamp(features['blur_score']/400.0,0.0,1.0); bright = clamp(1.0-abs(features['brightness_score']-150.0)/150.0,0.0,1.0); contrast = clamp(features['contrast_score']/80.0,0.0,1.0)
    score = 1.0 + (0.50*blur + 0.25*bright + 0.25*contrast) * 9.0
    return {'capture_quality_score':round(score,2)}

def compute_preliminary_gradix_score(capture_quality_score, geometry_features):
    capture = clamp((capture_quality_score-1.0)/9.0,0.0,1.0); coverage = clamp(geometry_features['coverage_ratio']/0.60,0.0,1.0); aspect = geometry_features['aspect_ratio_quality']; penalty = 0.20 if geometry_features['used_fallback'] else 0.0
    return {'gradix_preliminary_score':round(1.0 + clamp(0.55*capture + 0.30*coverage + 0.15*aspect - penalty, 0.0, 1.0)*9.0, 2)}

def compute_centering_score(f): return {'centering_score':round(1.0 + (0.5*f['horizontal_balance'] + 0.5*f['vertical_balance']) * 9.0, 2)}
def compute_edge_score(f): return {'gradix_edge_score':round(1.0 + clamp((f['edge_score']/100.0)*(0.75+0.25*clamp(f.get('edge_confidence',1.0),0.0,1.0)),0.0,1.0)*9.0, 2)}
def compute_corner_score(f): return {'gradix_corner_score':round(1.0 + clamp((f['corner_score_raw']/100.0)*(0.75+0.25*clamp(f.get('corner_confidence',1.0),0.0,1.0)),0.0,1.0)*9.0, 2)}
def compute_whitening_surface_score(f): return {'gradix_whitening_surface_score':round(1.0 + clamp((f['whitening_surface_score']/10.0)*(0.75+0.25*(0.5*clamp(f.get('whitening_confidence',1.0),0.0,1.0)+0.5*clamp(f.get('surface_confidence',1.0),0.0,1.0))),0.0,1.0)*9.0, 2)}

def compute_gradix_condition_stub_v4(preliminary_gradix_score, centering_score, gradix_edge_score, gradix_corner_score, gradix_whitening_surface_score, edge_confidence=1.0, corner_confidence=1.0, whitening_confidence=1.0, surface_confidence=1.0):
    prelim = clamp((preliminary_gradix_score-1.0)/9.0,0.0,1.0); cent = clamp((centering_score-1.0)/9.0,0.0,1.0); edge = clamp((gradix_edge_score-1.0)/9.0,0.0,1.0); corner = clamp((gradix_corner_score-1.0)/9.0,0.0,1.0); ws = clamp((gradix_whitening_surface_score-1.0)/9.0,0.0,1.0)
    ew = 0.25*(0.60+0.40*clamp(edge_confidence,0.0,1.0)); cw = 0.20*(0.60+0.40*clamp(corner_confidence,0.0,1.0)); ww = 0.15*(0.60+0.40*(0.5*clamp(whitening_confidence,0.0,1.0)+0.5*clamp(surface_confidence,0.0,1.0))); centw = 0.15; pw = 1.0-(ew+cw+ww+centw)
    return {'gradix_condition_stub_v4':round(1.0 + clamp(pw*prelim + centw*cent + ew*edge + cw*corner + ww*ws, 0.0, 1.0)*9.0, 2)}
'''),
code(r'''
%%writefile _gradix_local/pipeline.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import cv2, numpy as np, pandas as pd
from _gradix_local.utils import ensure_bgr_uint8, flatten_dict, infer_label_from_path, normalize_label_condition, target_from_label, normalize_to_0_1
from _gradix_local.vision import detect_card_contour, warp_card_perspective, validate_rectified_card
from _gradix_local.features import extract_visual_features, extract_geometry_features, extract_centering_features, draw_content_bbox
from _gradix_local.scoring import compute_capture_quality_score, compute_preliminary_gradix_score, compute_centering_score, compute_edge_score, compute_corner_score, compute_whitening_surface_score, compute_gradix_condition_stub_v4

FEATURE_SCHEMA_VERSION = 'gradix_pipeline_fuerte_notebook_v1'
VALID_EXTENSIONS = {'.jpg','.jpeg','.png','.webp'}

def _edge_and_corner_and_surface_modules(rectified_card_bgr):
    gray = cv2.cvtColor(rectified_card_bgr, cv2.COLOR_BGR2GRAY); h,w = gray.shape[:2]
    band = max(4,int(min(h,w)*0.03)); inset = max(1,int(min(h,w)*0.01)); regs = {'top':gray[inset:inset+band,inset:w-inset],'bottom':gray[h-inset-band:h-inset,inset:w-inset],'left':gray[inset:h-inset,inset:inset+band],'right':gray[inset:h-inset,w-inset-band:w-inset]}
    edge = {}; scores=[]; stds=[]
    for side, reg in regs.items():
        prof = reg.mean(axis=0 if side in {'top','bottom'} else 1).astype(np.float32); smooth = cv2.GaussianBlur(prof.reshape(-1,1),(1,11),0).ravel() if prof.size else prof; resid = np.abs(prof-smooth) if prof.size else np.array([0.0])
        anomaly = float(np.mean(resid > max(6.0, float(np.std(resid))*1.5))) if resid.size else 1.0; hi = float(np.mean(reg > (np.mean(reg)+max(10.0,np.std(reg)*1.2)))) if reg.size else 1.0; rough = float(np.std(np.diff(prof))) if prof.size > 1 else 0.0
        pen = 0.45*normalize_to_0_1(anomaly,0.02,0.22)+0.30*normalize_to_0_1(rough,2.5,18.0)+0.25*normalize_to_0_1(hi,0.01,0.18); score = 100.0*(1.0-max(0.0,min(1.0,pen)))
        edge[f'{side}_edge_score']=float(score); edge[f'{side}_anomaly_ratio']=float(anomaly); edge[f'{side}_highlight_ratio']=float(hi); edge[f'{side}_roughness']=float(rough); edge[f'{side}_std_intensity']=float(np.std(reg)); scores.append(score); stds.append(float(np.std(reg)))
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var()); contrast = float(np.std(gray)); avg = float(np.mean(stds)) if stds else 0.0; conf = max(0.0,min(1.0,0.45*normalize_to_0_1(lap,40,300)+0.30*normalize_to_0_1(contrast,20,70)+0.25*normalize_to_0_1(avg,8,35)))
    edge_score = float(np.mean(scores)) if scores else 0.0
    if conf < 0.20: edge_score *= 0.85
    elif conf < 0.35: edge_score *= 0.92
    edge['edge_score']=float(max(0.0,min(100.0,edge_score))); edge['edge_confidence']=float(conf)
    patch = max(16,int(min(h,w)*0.12)); patches = {'top_left':gray[inset:inset+patch,inset:inset+patch],'top_right':gray[inset:inset+patch,w-inset-patch:w-inset],'bottom_left':gray[h-inset-patch:h-inset,inset:inset+patch],'bottom_right':gray[h-inset-patch:h-inset,w-inset-patch:w-inset]}
    corner = {}; cs=[]; cstd=[]
    for name, patch_img in patches.items():
        g = cv2.GaussianBlur(patch_img,(3,3),0); m = float(np.mean(g)); s = float(np.std(g)); lapv = float(cv2.Laplacian(g, cv2.CV_64F).var()); hi = float(np.mean(g > (m+max(10.0,s*1.2))))
        pen = 0.35*normalize_to_0_1(hi,0.01,0.20)+0.45*normalize_to_0_1(lapv,35.0,550.0)+0.20*(normalize_to_0_1(max(0.0,8.0-s),0.0,8.0) if s < 8.0 else 0.0)
        score = 100.0*(1.0-max(0.0,min(1.0,pen))); corner[f'{name}_corner_score']=float(score); corner[f'{name}_highlight_ratio']=float(hi); corner[f'{name}_roughness']=float(lapv); corner[f'{name}_std_intensity']=float(s); cs.append(score); cstd.append(s)
    cconf = max(0.0,min(1.0,0.45*normalize_to_0_1(float(cv2.Laplacian(gray, cv2.CV_64F).var()),40,450)+0.30*normalize_to_0_1(float(np.std(gray)),20,70)+0.25*normalize_to_0_1(float(np.mean(cstd)) if cstd else 0.0,8,35)))
    raw = float(np.mean(cs)) if cs else 0.0
    if cconf < 0.20: raw *= 0.85
    elif cconf < 0.35: raw *= 0.92
    corner['corner_score_raw']=float(max(0.0,min(100.0,raw))); corner['corner_confidence']=float(cconf)
    hsv = cv2.cvtColor(rectified_card_bgr, cv2.COLOR_BGR2HSV); s,v = hsv[:,:,1], hsv[:,:,2]; bd = max(6,int(round(min(h,w)*0.025))); cp = max(12,int(round(min(h,w)*0.07)))
    edge_mask = np.zeros((h,w),dtype=np.uint8); edge_mask[:bd,:]=255; edge_mask[h-bd:,:]=255; edge_mask[:,:bd]=255; edge_mask[:,w-bd:]=255; corner_mask = np.zeros((h,w),dtype=np.uint8); corner_mask[:cp,:cp]=255; corner_mask[:cp,w-cp:]=255; corner_mask[h-cp:,:cp]=255; corner_mask[h-cp:,w-cp:]=255
    edge_wh = float(np.mean((v>=245)&(s<=35)&(edge_mask>0))); corner_wh = float(np.mean((v>=245)&(s<=35)&(corner_mask>0))); whitening_conf = float(max(0.0,min(1.0,0.55+0.20*min(1.0,bd/18.0)+0.25*min(1.0,cp/36.0))))
    inner = gray[int(h*0.12):int(h*0.88), int(w*0.12):int(w*0.88)]; glare = float(np.mean(inner>=245)) if inner.size else 0.0; texture = float(np.std(inner)/255.0) if inner.size else 0.0; dark = float(np.mean(inner<=35)) if inner.size else 0.0; surface_conf = float(max(0.0,min(1.0,0.60+0.20*min(1.0,(inner.shape[0] if inner.size else 0)/400.0)+0.20*min(1.0,(inner.shape[1] if inner.size else 0)/280.0))))
    whitening_score = max(0.0,min(10.0,10.0-(5.0*min(1.0,edge_wh/0.20)+5.0*min(1.0,corner_wh/0.25)))); surface_score = max(0.0,min(10.0,10.0-(4.0*min(1.0,glare/0.03)+3.5*min(1.0,texture/0.08)+2.5*min(1.0,dark/0.04)))); surface = {'whitening_score':float(whitening_score),'surface_score':float(surface_score),'whitening_surface_score':float(max(0.0,min(10.0,0.55*whitening_score+0.45*surface_score))),'edge_whitening_ratio':float(edge_wh),'corner_whitening_ratio':float(corner_wh),'whitening_confidence':float(whitening_conf),'glare_ratio':float(glare),'texture_anomaly_ratio':float(texture),'dark_spot_ratio':float(dark),'surface_confidence':float(surface_conf)}
    return edge, corner, surface

def analyze_card_image(image_bgr):
    image_bgr = ensure_bgr_uint8(image_bgr); detection = detect_card_contour(image_bgr); corners = detection.get('corners'); contour = detection.get('contour'); used_fallback = detection.get('used_fallback', True); debug = dict(detection.get('debug_images', {})); warped = None
    warp = {'computed':False,'reason':'missing_valid_corners','data':None}; postwarp = {'computed':False,'reason':'warp_not_available','data':None}; features = {'computed':False,'reason':'warp_not_available','visual':None,'geometry':None,'centering':None,'edge':None,'corner':None,'whitening_surface':None}; scores = {'computed':False,'reason':'features_not_available','capture_quality':None,'preliminary':None,'centering':None,'edge':None,'corner':None,'whitening_surface':None,'condition_stub_v4':None}; assessment = {'computed':False,'reason':'scores_not_available','capture_quality':None,'analysis_ready':False,'analysis_recommended':False}
    if corners is not None and len(corners) == 4:
        warp_result = warp_card_perspective(image_bgr, corners); warped = warp_result.get('warped_image'); warp = {'computed':True,'reason':'','data':warp_result}
    if warped is not None:
        post = validate_rectified_card(warped); postwarp = {'computed':True,'reason':'','data':post}; visual = extract_visual_features(warped); capture = compute_capture_quality_score(visual); geometry = extract_geometry_features(contour, image_bgr.shape, visual['aspect_ratio'], used_fallback); centering = extract_centering_features(warped); edge, corner, surface = _edge_and_corner_and_surface_modules(warped); features = {'computed':True,'reason':'','visual':visual,'geometry':geometry,'centering':centering,'edge':edge,'corner':corner,'whitening_surface':surface}; prelim = compute_preliminary_gradix_score(capture['capture_quality_score'], geometry); cent_score = compute_centering_score(centering); edge_score = compute_edge_score(edge); corner_score = compute_corner_score(corner); ws_score = compute_whitening_surface_score(surface); stub = compute_gradix_condition_stub_v4(prelim['gradix_preliminary_score'], cent_score['centering_score'], edge_score['gradix_edge_score'], corner_score['gradix_corner_score'], ws_score['gradix_whitening_surface_score'], edge['edge_confidence'], corner['corner_confidence'], surface['whitening_confidence'], surface['surface_confidence']); scores = {'computed':True,'reason':'','capture_quality':capture,'preliminary':prelim,'centering':cent_score,'edge':edge_score,'corner':corner_score,'whitening_surface':ws_score,'condition_stub_v4':stub}
        coverage = 0.0 if contour is None else float(abs(cv2.contourArea(contour)) / max(1.0, image_bgr.shape[0]*image_bgr.shape[1])); aspect_ok = 0.65 <= visual['aspect_ratio'] <= 0.78
        if used_fallback: cap_level, cap_msg = 'deficiente', 'La deteccion uso un contorno de respaldo.'
        elif coverage > 0.95: cap_level, cap_msg = ('mejorable' if capture['capture_quality_score'] >= 5.0 else 'deficiente'), 'La carta no tiene margenes visibles.'
        elif aspect_ok and coverage >= 0.35 and capture['capture_quality_score'] >= 7.0: cap_level, cap_msg = 'buena', 'La captura es adecuada para analisis preliminar.'
        elif aspect_ok and coverage >= 0.22 and capture['capture_quality_score'] >= 5.0: cap_level, cap_msg = 'mejorable', 'La captura puede procesarse, pero conviene mejorarla.'
        else: cap_level, cap_msg = 'deficiente', 'La captura no es ideal para analisis confiable.'
        assessment = {'computed':True,'reason':'','capture_quality':{'coverage_ratio':round(coverage,3),'aspect_ok':aspect_ok,'used_fallback':bool(used_fallback),'capture_assessment':cap_level,'capture_message':cap_msg},'analysis_ready':bool(post['postwarp_valid']),'analysis_recommended':bool(not post['retry_recommended'])}
        debug['warped_card']=warped; debug['warped_with_bbox']=draw_content_bbox(warped, centering['content_bbox'])
    return {'detection':detection,'warp':warp,'postwarp_validation':postwarp,'features':features,'scores':scores,'assessment':assessment,'debug_images':debug}

def evaluate_analysis_status(used_fallback, det, post, visual, geometry, edge, corner):
    invalid=[]; warn=[]; best=float(det.get('best_score',0.0)); conf=float(det.get('detection_confidence',0.0)); weak=bool(det.get('weak_detection',False)); coverage=float(geometry.get('coverage_ratio',0.0)); blur=float(visual.get('blur_score',0.0)); bright=float(visual.get('brightness_score',0.0)); contrast=float(visual.get('contrast_score',0.0)); edge_conf=float(edge.get('edge_confidence',0.0)); corner_conf=float(corner.get('corner_confidence',0.0)); post_valid=bool(post.get('postwarp_valid',False)); post_score=float(post.get('postwarp_score',0.0)); retry=bool(post.get('retry_recommended',False))
    if used_fallback and best < 0.50: invalid.append('fallback_detection')
    elif used_fallback: warn.append('fallback_used')
    if weak and conf < 0.45: invalid.append('weak_detection')
    elif weak: warn.append('weak_detection')
    if coverage < 0.15: invalid.append('low_coverage')
    if coverage > 0.99: invalid.append('excessive_coverage')
    if blur < 70: invalid.append('blurry_image')
    if bright < 75: invalid.append('too_dark')
    if bright > 235: invalid.append('too_bright')
    if contrast < 20: invalid.append('low_contrast')
    if edge_conf < 0.45: invalid.append('low_edge_confidence')
    if corner_conf < 0.45: invalid.append('low_corner_confidence')
    if (not post_valid) and post_score < 0.45: invalid.append('invalid_postwarp')
    elif (not post_valid) or retry: warn.append('weak_postwarp')
    if invalid: return {'analysis_status':'invalid_capture','invalid_reasons':'|'.join(sorted(set(invalid))),'usable_for_condition_model':False}
    if 0.28 <= coverage < 0.38: warn.append('borderline_low_coverage')
    if coverage > 0.90: warn.append('borderline_high_coverage')
    if 70 <= blur < 120: warn.append('moderate_blur')
    if 75 <= bright < 95: warn.append('suboptimal_dark_brightness')
    if 210 < bright <= 235: warn.append('suboptimal_high_brightness')
    if 20 <= contrast < 30: warn.append('suboptimal_contrast')
    if 0.45 <= edge_conf < 0.60: warn.append('moderate_edge_confidence')
    if 0.45 <= corner_conf < 0.60: warn.append('moderate_corner_confidence')
    if 0.45 <= conf < 0.60: warn.append('moderate_detection_confidence')
    if 0.45 <= post_score < 0.60: warn.append('moderate_postwarp_score')
    if warn: return {'analysis_status':'valid_with_warning','invalid_reasons':'|'.join(sorted(set(warn))),'usable_for_condition_model':False}
    return {'analysis_status':'valid','invalid_reasons':'','usable_for_condition_model':True}
'''),
]
cells += [
code(r'''
%%writefile _gradix_local/dataset_builder.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import cv2, pandas as pd
from _gradix_local.utils import flatten_dict, infer_label_from_path, normalize_label_condition, target_from_label
from _gradix_local.pipeline import analyze_card_image, evaluate_analysis_status, FEATURE_SCHEMA_VERSION, VALID_EXTENSIONS

def process_image_file(image_path, raw_root):
    image_path = Path(image_path); raw_root = Path(raw_root); label_raw = infer_label_from_path(image_path, raw_root); label = normalize_label_condition(label_raw); target = target_from_label(label)
    row = {'image_id':image_path.stem,'image_filename':image_path.name,'image_path':str(image_path),'relative_path_from_raw':str(image_path.relative_to(raw_root)),'categoria_carpeta_raw':label_raw,'label_condition':label,'target_damaged':target,'run_timestamp':datetime.utcnow().isoformat(),'feature_schema_version':FEATURE_SCHEMA_VERSION,'procesado_exito':False,'analysis_status':'not_evaluated','invalid_reasons':'','usable_for_condition_model':False,'pipeline_stage':'init','error':''}
    image = cv2.imread(str(image_path))
    if image is None: row.update({'pipeline_stage':'read_error','error':'No se pudo leer la imagen','analysis_status':'invalid_capture','invalid_reasons':'read_error'}); return row, {'debug_images':{},'blocks':{}}
    row['image_height']=int(image.shape[0]); row['image_width']=int(image.shape[1])
    try:
        analysis = analyze_card_image(image); det=analysis['detection']; warp=analysis['warp']; post=analysis['postwarp_validation']; feats=analysis['features']; scores=analysis['scores']; assess=analysis['assessment']; corners = det.get('corners'); used_fallback = det.get('used_fallback',False); detm = det.get('metrics',{}); warp_data = warp.get('data') if warp.get('computed') else None; post_data = post.get('data') if post.get('computed') else {}; visual = feats.get('visual') or {}; geometry = feats.get('geometry') or {}; centering = feats.get('centering') or {}; edge = feats.get('edge') or {}; corner = feats.get('corner') or {}; surface = feats.get('whitening_surface') or {}
        row.update({'det_success':bool(det.get('success',False)),'det_used_fallback':bool(used_fallback),'warp_success':bool(warp_data and warp_data.get('warped_image') is not None),'postwarp_computed':bool(post.get('computed',False)),'features_computed':bool(feats.get('computed',False)),'scores_computed':bool(scores.get('computed',False))})
        row.update(flatten_dict(detm,'det')); row.update(flatten_dict(warp_data.get('metrics',{}),'warp') if warp_data is not None else {}); row.update(flatten_dict(post_data,'postwarp')); row.update(flatten_dict(visual,'visual')); row.update(flatten_dict(geometry,'geometry')); row.update(flatten_dict(centering,'centrado')); row.update(flatten_dict(edge,'borde')); row.update(flatten_dict(corner,'esquina')); row.update(flatten_dict(surface,'superficie')); row.update(flatten_dict(scores.get('capture_quality') or {},'score_capture')); row.update(flatten_dict(scores.get('preliminary') or {},'score_prelim')); row.update(flatten_dict(scores.get('centering') or {},'score_centering')); row.update(flatten_dict(scores.get('edge') or {},'score_edge')); row.update(flatten_dict(scores.get('corner') or {},'score_corner')); row.update(flatten_dict(scores.get('whitening_surface') or {},'score_ws')); row.update(flatten_dict(scores.get('condition_stub_v4') or {},'score_stub_v4')); row.update(flatten_dict(assess.get('capture_quality') or {},'capture_assessment'))
        if corners is None or len(corners) != 4: row.update({'pipeline_stage':'detection_failed','error':'No se detectaron 4 esquinas validas','analysis_status':'invalid_capture','invalid_reasons':'invalid_detection'}); return row, {'debug_images':analysis['debug_images'],'blocks':analysis}
        if not row['warp_success']: row.update({'pipeline_stage':'warp_failed','error':'No se pudo rectificar la carta','analysis_status':'invalid_capture','invalid_reasons':'warp_failed'}); return row, {'debug_images':analysis['debug_images'],'blocks':analysis}
        if not feats.get('computed',False): row.update({'pipeline_stage':'features_failed','error':feats.get('reason','No se pudieron calcular features'),'analysis_status':'invalid_capture','invalid_reasons':'features_not_available'}); return row, {'debug_images':analysis['debug_images'],'blocks':analysis}
        row.update(evaluate_analysis_status(used_fallback, detm, post_data, visual, geometry, edge, corner)); row['procesado_exito']=True; row['pipeline_stage']='completed'; return row, {'debug_images':analysis['debug_images'],'blocks':analysis}
    except Exception as exc:
        row.update({'pipeline_stage':'exception','error':str(exc),'analysis_status':'invalid_capture','invalid_reasons':'exception'}); return row, {'debug_images':{},'blocks':{'exception':str(exc)}}

def process_image_batch(raw_root, output_csv=None, verbose=True):
    raw_root = Path(raw_root); files = sorted([p for p in raw_root.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]); rows=[]; example={}
    if verbose: print(f'Procesando {len(files)} imagenes desde {raw_root}')
    for idx, path in enumerate(files,1):
        row, artifacts = process_image_file(path, raw_root); rows.append(row)
        if idx == 1: example = {'image_path':path, **artifacts}
        if verbose and (idx % 25 == 0 or idx == len(files)): print(f'  {idx}/{len(files)}')
    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True); df.to_csv(output_csv, index=False)
    return df, example
'''),
code('''
from _gradix_local.dataset_builder import FEATURE_SCHEMA_VERSION, process_image_file, process_image_batch
from _gradix_local.utils import infer_label_from_path, normalize_label_condition, target_from_label
print('Pipeline local listo:', FEATURE_SCHEMA_VERSION)
'''),
md('''
# Descripcion De Los Datos

El conjunto de datos parte de **capturas propias de cartas** organizadas manualmente en `data/raw`. Esta organizacion cumple dos funciones metodologicas. Primero, define la unidad observacional: cada archivo de imagen representa una captura individual. Segundo, aporta la etiqueta inicial de condicion a partir de la carpeta padre, lo que permite construir un target supervisado reproducible sin depender de anotacion embebida en metadatos.

La estructura esperada es de la forma `data/raw/<clase>/<archivo>`. En este notebook, `label_condition` se obtiene a partir del nombre de la carpeta y luego se normaliza a un conjunto controlado de etiquetas. A partir de esa etiqueta normalizada se construye `target_damaged`, donde `1` representa `damaged` y `0` representa `undamaged`. Este target es **operacional**: sirve para entrenar modelos de clasificacion, pero no sustituye una certificacion profesional de grading.
'''),
code('''
raw_image_paths = sorted([p for p in INPUT_DIR.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])
raw_records = []
for path in raw_image_paths:
    label_raw = infer_label_from_path(path, INPUT_DIR)
    label_condition = normalize_label_condition(label_raw)
    target_damaged = target_from_label(label_condition)
    raw_records.append({
        'image_filename': path.name,
        'relative_path_from_raw': str(path.relative_to(INPUT_DIR)),
        'folder_label_raw': label_raw,
        'label_condition': label_condition,
        'target_damaged': target_damaged,
    })
raw_overview_df = pd.DataFrame(raw_records)

print('Numero total de imagenes detectadas:', len(raw_overview_df))
if not raw_overview_df.empty:
    print('\\nConteo por clase normalizada:')
    display(raw_overview_df['label_condition'].value_counts(dropna=False).rename_axis('label_condition').reset_index(name='count'))
    print('\\nEjemplos de estructura de data/raw:')
    display(raw_overview_df.head(10))
else:
    print('No se encontraron imagenes en data/raw.')
'''),
md('''
# Ejemplo Unitario Con Una Imagen
'''),
code('''
def show_images(items, cols=3, figsize=(16,10)):
    if not items:
        print('No hay imagenes para mostrar.'); return
    rows = int(np.ceil(len(items)/cols)); fig, axes = plt.subplots(rows, cols, figsize=figsize); axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.ravel(): ax.axis('off')
    for ax, (title, image) in zip(axes.ravel(), items):
        ax.imshow(image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray' if image.ndim == 2 else None)
        ax.set_title(title); ax.axis('off')
    plt.tight_layout(); plt.show()

def sanitize_for_display(value):
    if isinstance(value, np.ndarray): return {'type':'ndarray','shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value, dict): return {k:sanitize_for_display(v) for k,v in value.items()}
    if isinstance(value, list): return [sanitize_for_display(v) for v in value]
    return value

image_paths = sorted([p for p in INPUT_DIR.rglob('*') if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])
if not image_paths: raise RuntimeError('No se encontraron imagenes en INPUT_DIR.')
sample_path = image_paths[0]
sample_row, sample_artifacts = process_image_file(sample_path, INPUT_DIR)
print('Imagen de ejemplo:', sample_path)
display(pd.Series(sample_row).head(40))
'''),
code('''
sample_debug = sample_artifacts.get('debug_images', {})
original = cv2.imread(str(sample_path))
items = [('original', original)]
for key in ['gray','canny_closed','structural_mask','multilayer_overlay','detected_contour_overlay','warped_card','warped_with_bbox']:
    if key in sample_debug: items.append((key, sample_debug[key]))
show_images(items, cols=3, figsize=(17,11))
'''),
code('''
blocks = sample_artifacts.get('blocks', {})
for block_name in ['detection','warp','postwarp_validation','features','scores','assessment']:
    if block_name in blocks:
        print(f'\\n{block_name.upper()}')
        display(pd.Series(sanitize_for_display(blocks[block_name])))
'''),
md('''
# Generacion Del Dataset Completo
'''),
code('''
df_dataset, first_example = process_image_batch(INPUT_DIR, output_csv=OUTPUT_CSV, verbose=True)
print('Dataset exportado en:', OUTPUT_CSV)
print('Shape:', df_dataset.shape)
display(df_dataset.head())
'''),
md('''
# EDA

El analisis exploratorio del dataset cumple dos objetivos. El primero es describir la distribucion de etiquetas y estados de procesamiento. El segundo es inspeccionar si las variables tecnicas extraidas por el pipeline presentan rangos y dispersiones coherentes con el problema. Esto ayuda a defender que el dataset no surge de una caja negra, sino de un proceso controlado y auditable.

En particular, se revisan distribuciones de clase, resultados del control tecnico (`analysis_status`, `usable_for_condition_model`) y algunas variables continuas relevantes como nitidez, brillo, contraste y cobertura geometrica. Tambien se examinan las razones de invalidez para entender por que una captura fue descartada o marcada con advertencia.
'''),
code('''
def find_existing_column(df, candidates):
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    return None

for column_name in ['label_condition', 'target_damaged', 'analysis_status', 'usable_for_condition_model']:
    if column_name in df_dataset.columns:
        print(f'\\nvalue_counts de {column_name}:')
        display(df_dataset[column_name].value_counts(dropna=False).rename_axis(column_name).reset_index(name='count'))

histogram_candidates = {
    'blur_score': ['visual_blur_score', 'blur_score'],
    'brightness_score': ['visual_brightness_score', 'brightness_score'],
    'contrast_score': ['visual_contrast_score', 'contrast_score'],
    'geometry_coverage_ratio': ['geometry_coverage_ratio', 'coverage_ratio'],
}

available_histograms = [(label, find_existing_column(df_dataset, candidates)) for label, candidates in histogram_candidates.items()]
available_histograms = [(label, column_name) for label, column_name in available_histograms if column_name is not None]

if available_histograms:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (label, column_name) in zip(axes, available_histograms):
        series = pd.to_numeric(df_dataset[column_name], errors='coerce').dropna()
        ax.hist(series, bins=30, color='#4C78A8', edgecolor='white')
        ax.set_title(f'Histograma de {label}')
        ax.set_xlabel(column_name)
        ax.set_ylabel('Frecuencia')
        ax.axis('on')
    plt.tight_layout()
    plt.show()

invalid_reason_series = (
    df_dataset['invalid_reasons']
    .fillna('')
    .astype(str)
    .str.split('|')
    .explode()
    .str.strip()
)
invalid_reason_series = invalid_reason_series[invalid_reason_series.ne('')]
print('\\nTop invalid_reasons:')
display(invalid_reason_series.value_counts().head(15).rename_axis('invalid_reason').reset_index(name='count'))

valid_examples = df_dataset[df_dataset['usable_for_condition_model'] == True].head(5) if 'usable_for_condition_model' in df_dataset.columns else pd.DataFrame()
invalid_examples = df_dataset[df_dataset['analysis_status'] == 'invalid_capture'].head(5) if 'analysis_status' in df_dataset.columns else pd.DataFrame()

example_columns = [c for c in ['image_filename', 'label_condition', 'target_damaged', 'analysis_status', 'invalid_reasons', 'usable_for_condition_model'] if c in df_dataset.columns]
print('\\nEjemplos de filas validas tecnicamente:')
display(valid_examples[example_columns] if not valid_examples.empty else pd.DataFrame(columns=example_columns))
print('\\nEjemplos de filas invalidas:')
display(invalid_examples[example_columns] if not invalid_examples.empty else pd.DataFrame(columns=example_columns))
'''),
md('''
# Preparacion Y Limpieza De Datos

La preparacion de datos en Gradix no se limita a convertir archivos en filas. Tambien implica **normalizar etiquetas**, **construir el target supervisado** y **filtrar observaciones tecnicamente aptas**. En otras palabras, la limpieza no se interpreta como un paso cosmetico, sino como una etapa de control de calidad del proceso de captura y analisis.

Primero, la etiqueta de origen se toma de la carpeta padre y se normaliza hacia un vocabulario reducido: `damaged`, `undamaged` o `unknown`. Segundo, a partir de esa etiqueta se construye `target_damaged`. Tercero, el pipeline evalua la calidad tecnica de cada captura. Una imagen puede terminar como `invalid_capture` cuando la deteccion, el warp o las evidencias visuales son insuficientes. Puede terminar como `valid_with_warning` cuando el procesamiento es posible, pero hay señales de riesgo tecnico. Solo las filas con `usable_for_condition_model = True` se consideran adecuadas como insumo limpio para modelacion supervisada.

Esta separacion es importante metodologicamente: una imagen puede tener etiqueta de clase, pero no por ello ser una observacion confiable para entrenar un modelo. El campo `usable_for_condition_model` funciona entonces como un **filtro tecnico explicito** entre disponibilidad de datos y aptitud real para modelado.
'''),
code('''
folder_label_examples = pd.DataFrame({
    'folder_label_raw': sorted(raw_overview_df['folder_label_raw'].dropna().astype(str).unique()) if not raw_overview_df.empty else [],
})
if not folder_label_examples.empty:
    folder_label_examples['label_condition_normalized'] = folder_label_examples['folder_label_raw'].map(normalize_label_condition)
    folder_label_examples['target_damaged'] = folder_label_examples['label_condition_normalized'].map(target_from_label)
print('Normalizacion de etiquetas observada en data/raw:')
display(folder_label_examples if not folder_label_examples.empty else pd.DataFrame(columns=['folder_label_raw', 'label_condition_normalized', 'target_damaged']))

criteria_table = pd.DataFrame([
    {'grupo': 'invalid_capture', 'criterio': 'Deteccion debil o por fallback con score bajo', 'referencia': 'fallback_detection, weak_detection'},
    {'grupo': 'invalid_capture', 'criterio': 'Cobertura geometrica extrema o insuficiente', 'referencia': 'low_coverage, excessive_coverage'},
    {'grupo': 'invalid_capture', 'criterio': 'Problemas de calidad visual severos', 'referencia': 'blurry_image, too_dark, too_bright, low_contrast'},
    {'grupo': 'invalid_capture', 'criterio': 'Confianza insuficiente en bordes o esquinas', 'referencia': 'low_edge_confidence, low_corner_confidence'},
    {'grupo': 'invalid_capture', 'criterio': 'Post-warp claramente invalido', 'referencia': 'invalid_postwarp'},
    {'grupo': 'valid_with_warning', 'criterio': 'Uso de fallback o deteccion debil no critica', 'referencia': 'fallback_used, weak_detection'},
    {'grupo': 'valid_with_warning', 'criterio': 'Cobertura, blur o brillo en rango limite', 'referencia': 'borderline_low_coverage, moderate_blur, suboptimal_*'},
    {'grupo': 'valid_with_warning', 'criterio': 'Confianzas moderadas o post-warp intermedio', 'referencia': 'moderate_*'},
    {'grupo': 'usable_for_condition_model', 'criterio': 'Solo filas sin invalidaciones ni advertencias', 'referencia': 'analysis_status = valid'},
])
print('\\nResumen de criterios tecnicos del pipeline:')
display(criteria_table)

cleaning_summary = {
    'total_rows': int(len(df_dataset)),
    'processed_successfully': int(df_dataset['procesado_exito'].fillna(False).sum()) if 'procesado_exito' in df_dataset.columns else None,
    'rows_with_known_target': int(df_dataset['target_damaged'].notna().sum()) if 'target_damaged' in df_dataset.columns else None,
    'rows_invalid_capture': int((df_dataset['analysis_status'] == 'invalid_capture').sum()) if 'analysis_status' in df_dataset.columns else None,
    'rows_valid_with_warning': int((df_dataset['analysis_status'] == 'valid_with_warning').sum()) if 'analysis_status' in df_dataset.columns else None,
    'rows_valid': int((df_dataset['analysis_status'] == 'valid').sum()) if 'analysis_status' in df_dataset.columns else None,
    'rows_usable_for_condition_model': int((df_dataset['usable_for_condition_model'] == True).sum()) if 'usable_for_condition_model' in df_dataset.columns else None,
}
print('\\nResumen de limpieza y filtrado:')
display(pd.Series(cleaning_summary))

if 'usable_for_condition_model' in df_dataset.columns:
    filtered_for_model_df = df_dataset[df_dataset['usable_for_condition_model'] == True].copy()
else:
    filtered_for_model_df = df_dataset.copy()
print('\\nShape del dataset completo:', df_dataset.shape)
print('Shape del subconjunto tecnicamente utilizable:', filtered_for_model_df.shape)
'''),
md('''
# Resumen Estadistico Del Dataset
'''),
code('''
print(df_dataset.shape)
print(df_dataset.columns.tolist()[:100])
for column_name in ['label_condition','target_damaged','analysis_status','usable_for_condition_model']:
    if column_name in df_dataset.columns:
        print(f'\\nvalue_counts de {column_name}:')
        display(df_dataset[column_name].value_counts(dropna=False).rename_axis(column_name).reset_index(name='count'))
dataset_report = {'total_rows':int(len(df_dataset)),'total_columns':int(len(df_dataset.columns)),'label_distribution':df_dataset['label_condition'].value_counts(dropna=False).to_dict() if 'label_condition' in df_dataset.columns else {},'analysis_status_distribution':df_dataset['analysis_status'].value_counts(dropna=False).to_dict() if 'analysis_status' in df_dataset.columns else {},'usable_for_condition_model_distribution':df_dataset['usable_for_condition_model'].value_counts(dropna=False).to_dict() if 'usable_for_condition_model' in df_dataset.columns else {},'feature_schema_version':FEATURE_SCHEMA_VERSION}
dataset_report
'''),
md('''
# Cierre Metodologico

Quedaron integradas en el notebook las partes principales del pipeline real o casi real de Gradix: utilidades, preprocesado, deteccion fuerte, warp, validacion post-warp, features, scoring, evaluacion del analisis y procesamiento por lote.

Quedaron excluidas deliberadamente `app.py`, Streamlit, OCR, TCGdex, OpenAI y servicios externos porque no pertenecen al nucleo local de construccion del dataset.
''')
]

nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print(out)

