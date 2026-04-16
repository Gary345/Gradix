import os
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import numpy as np

from src.pipeline.card_analysis import analyze_card_image

FEATURE_SCHEMA_VERSION = "gradix_features_v2"


def aplanar_diccionario(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(aplanar_diccionario(v, new_key, sep=sep).items())
        elif isinstance(v, (int, float, str, bool, np.integer, np.floating, np.bool_)):
            items.append((new_key, v))
        elif v is None:
            items.append((new_key, None))

    return dict(items)


def obtener_label_desde_raw(ruta_imagen: Path, carpeta_raw: Path) -> str:
    try:
        relativa = ruta_imagen.relative_to(carpeta_raw)
        partes = relativa.parts
        if len(partes) >= 2:
            return partes[0].lower().strip()
    except Exception:
        pass
    return "unknown"


def normalizar_label_condition(label: str) -> str:
    label = (label or "").lower().strip()

    if label in {"damaged", "damage", "rotas", "rota", "danada", "da\u00f1ada"}:
        return "damaged"
    if label in {"undamaged", "clean", "sana", "sin_danio", "sin_da\u00f1o", "ok"}:
        return "undamaged"
    return "unknown"


def target_desde_label(label: str):
    if label == "damaged":
        return 1
    if label == "undamaged":
        return 0
    return None


def evaluar_estado_analisis(
    used_fallback: bool,
    detection_metrics: dict,
    postwarp: dict,
    visuales: dict,
    geometry: dict,
    bordes: dict,
    esquinas: dict,
) -> dict:
    invalid_reasons = []
    warning_reasons = []

    best_score = float(detection_metrics.get("best_score", 0.0))
    detection_confidence = float(detection_metrics.get("detection_confidence", 0.0))
    weak_detection = bool(detection_metrics.get("weak_detection", False))
    coverage_ratio = float(geometry.get("coverage_ratio", 0.0))

    blur_score = float(visuales.get("blur_score", 0.0))
    brightness_score = float(visuales.get("brightness_score", 0.0))
    contrast_score = float(visuales.get("contrast_score", 0.0))

    edge_confidence = float(bordes.get("edge_confidence", 0.0))
    corner_confidence = float(esquinas.get("corner_confidence", 0.0))

    postwarp_valid = bool(postwarp.get("postwarp_valid", False))
    postwarp_score = float(postwarp.get("postwarp_score", 0.0))
    retry_recommended = bool(postwarp.get("retry_recommended", False))

    if used_fallback and best_score < 0.50:
        invalid_reasons.append("fallback_detection")
    elif used_fallback:
        warning_reasons.append("fallback_used")

    if weak_detection and detection_confidence < 0.45:
        invalid_reasons.append("weak_detection")
    elif weak_detection:
        warning_reasons.append("weak_detection")

    if coverage_ratio < 0.15:
        invalid_reasons.append("low_coverage")
    if coverage_ratio > 0.99:
        invalid_reasons.append("excessive_coverage")
    if blur_score < 70:
        invalid_reasons.append("blurry_image")
    if brightness_score < 75:
        invalid_reasons.append("too_dark")
    if brightness_score > 235:
        invalid_reasons.append("too_bright")
    if contrast_score < 20:
        invalid_reasons.append("low_contrast")
    if edge_confidence < 0.45:
        invalid_reasons.append("low_edge_confidence")
    if corner_confidence < 0.45:
        invalid_reasons.append("low_corner_confidence")

    if not postwarp_valid and postwarp_score < 0.45:
        invalid_reasons.append("invalid_postwarp")
    elif not postwarp_valid or retry_recommended:
        warning_reasons.append("weak_postwarp")

    if invalid_reasons:
        return {
            "analysis_status": "invalid_capture",
            "invalid_reasons": "|".join(sorted(set(invalid_reasons))),
            "usable_for_condition_model": False,
        }

    if 0.28 <= coverage_ratio < 0.38:
        warning_reasons.append("borderline_low_coverage")
    if coverage_ratio > 0.90:
        warning_reasons.append("borderline_high_coverage")
    if 70 <= blur_score < 120:
        warning_reasons.append("moderate_blur")
    if 75 <= brightness_score < 95:
        warning_reasons.append("suboptimal_dark_brightness")
    if 210 < brightness_score <= 235:
        warning_reasons.append("suboptimal_high_brightness")
    if 20 <= contrast_score < 30:
        warning_reasons.append("suboptimal_contrast")
    if 0.45 <= edge_confidence < 0.60:
        warning_reasons.append("moderate_edge_confidence")
    if 0.45 <= corner_confidence < 0.60:
        warning_reasons.append("moderate_corner_confidence")
    if 0.45 <= detection_confidence < 0.60:
        warning_reasons.append("moderate_detection_confidence")
    if 0.45 <= postwarp_score < 0.60:
        warning_reasons.append("moderate_postwarp_score")

    if warning_reasons:
        return {
            "analysis_status": "valid_with_warning",
            "invalid_reasons": "|".join(sorted(set(warning_reasons))),
            "usable_for_condition_model": False,
        }

    return {
        "analysis_status": "valid",
        "invalid_reasons": "",
        "usable_for_condition_model": True,
    }


def procesar_lote_imagenes(carpeta_imagenes: str, archivo_salida_base: str):
    carpeta = Path(carpeta_imagenes)
    salida_base = Path(archivo_salida_base)

    extensiones_validas = {".jpg", ".jpeg", ".png", ".webp"}
    archivos = sorted(
        [p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in extensiones_validas]
    )

    datos_cartas = []
    print(f"Iniciando procesamiento de {len(archivos)} imagenes en subcarpetas de '{carpeta}'...\n")

    for indice, ruta_imagen in enumerate(archivos, 1):
        nombre_archivo = ruta_imagen.name
        categoria_raw = obtener_label_desde_raw(ruta_imagen, carpeta)
        label_condition = normalizar_label_condition(categoria_raw)
        target_damaged = target_desde_label(label_condition)

        print(
            f"[{indice}/{len(archivos)}] Procesando: {nombre_archivo} "
            f"(label={label_condition}) ..."
        )

        fila = {
            "image_id": ruta_imagen.stem,
            "image_filename": ruta_imagen.name,
            "image_path": str(ruta_imagen),
            "relative_path_from_raw": str(ruta_imagen.relative_to(carpeta)),
            "categoria_carpeta_raw": categoria_raw,
            "label_condition": label_condition,
            "target_damaged": target_damaged,
            "run_timestamp": datetime.utcnow().isoformat(),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "procesado_exito": False,
            "pipeline_stage": "init",
            "error": "",
            "analysis_status": "not_evaluated",
            "invalid_reasons": "",
            "usable_for_condition_model": False,
        }

        imagen = cv2.imread(str(ruta_imagen))
        if imagen is None:
            fila["pipeline_stage"] = "read_error"
            fila["error"] = "No se pudo leer la imagen"
            datos_cartas.append(fila)
            print("  -> Error: No se pudo leer la imagen")
            continue

        fila["image_height"] = int(imagen.shape[0])
        fila["image_width"] = int(imagen.shape[1])

        try:
            analysis = analyze_card_image(imagen)

            deteccion = analysis["detection"]
            warp_block = analysis["warp"]
            postwarp_block = analysis["postwarp_validation"]
            features_block = analysis["features"]
            scores_block = analysis["scores"]
            assessment_block = analysis["assessment"]

            corners = deteccion.get("corners")
            used_fallback = deteccion.get("used_fallback", False)
            detection_metrics = deteccion.get("metrics", {})

            warp_data = warp_block.get("data") if warp_block.get("computed") else None
            postwarp_data = postwarp_block.get("data") if postwarp_block.get("computed") else {}

            visuales = features_block.get("visual") or {}
            geometry = features_block.get("geometry") or {}
            centrado = features_block.get("centering") or {}
            bordes = features_block.get("edge") or {}
            esquinas = features_block.get("corner") or {}
            superficie = features_block.get("whitening_surface") or {}

            capture_score = scores_block.get("capture_quality") or {}
            preliminary_score = scores_block.get("preliminary") or {}
            centering_score = scores_block.get("centering") or {}
            edge_score = scores_block.get("edge") or {}
            corner_score = scores_block.get("corner") or {}
            ws_score = scores_block.get("whitening_surface") or {}
            stub_v1 = scores_block.get("condition_stub_v1") or {}
            stub_v2 = scores_block.get("condition_stub_v2") or {}
            stub_v3 = scores_block.get("condition_stub_v3") or {}
            stub_v4 = scores_block.get("condition_stub_v4") or {}
            capture_assessment = assessment_block.get("capture_quality") or {}

            fila["det_success"] = bool(deteccion.get("success", False))
            fila["det_used_fallback"] = bool(used_fallback)
            fila["warp_success"] = bool(warp_data and warp_data.get("warped_image") is not None)
            fila["postwarp_computed"] = bool(postwarp_block.get("computed", False))
            fila["features_computed"] = bool(features_block.get("computed", False))
            fila["scores_computed"] = bool(scores_block.get("computed", False))

            fila.update(aplanar_diccionario(detection_metrics, parent_key="det"))
            if warp_data is not None:
                fila.update(aplanar_diccionario(warp_data.get("metrics", {}), parent_key="warp"))
            postwarp_flat = dict(postwarp_data)
            postwarp_flat.pop("postwarp_valid", None)
            postwarp_flat.pop("postwarp_score", None)
            postwarp_flat.pop("retry_recommended", None)
            fila.update(aplanar_diccionario(postwarp_flat, parent_key="postwarp"))
            fila.update(aplanar_diccionario(visuales, parent_key="visual"))
            fila.update(aplanar_diccionario(geometry, parent_key="geometry"))
            fila.update(aplanar_diccionario(centrado, parent_key="centrado"))
            fila.update(aplanar_diccionario(bordes, parent_key="borde"))
            fila.update(aplanar_diccionario(esquinas, parent_key="esquina"))
            fila.update(aplanar_diccionario(superficie, parent_key="superficie"))
            fila.update(aplanar_diccionario(capture_score, parent_key="score_capture"))
            fila.update(aplanar_diccionario(preliminary_score, parent_key="score_prelim"))
            fila.update(aplanar_diccionario(centering_score, parent_key="score_centering"))
            fila.update(aplanar_diccionario(edge_score, parent_key="score_edge"))
            fila.update(aplanar_diccionario(corner_score, parent_key="score_corner"))
            fila.update(aplanar_diccionario(ws_score, parent_key="score_ws"))
            fila.update(aplanar_diccionario(stub_v1, parent_key="score_stub_v1"))
            fila.update(aplanar_diccionario(stub_v2, parent_key="score_stub_v2"))
            fila.update(aplanar_diccionario(stub_v3, parent_key="score_stub_v3"))
            fila.update(aplanar_diccionario(stub_v4, parent_key="score_stub_v4"))
            fila.update(aplanar_diccionario(capture_assessment, parent_key="capture_assessment"))

            izq = centrado.get("left_margin", 0)
            der = centrado.get("right_margin", 0)
            if (izq + der) > 0:
                fila["ratio_centrado_calculado"] = round(izq / (izq + der), 3)

            fila["det_detection_confidence"] = float(detection_metrics.get("detection_confidence", 0.0))
            fila["det_weak_detection"] = bool(detection_metrics.get("weak_detection", False))
            fila["postwarp_valid"] = bool(postwarp_data.get("postwarp_valid", False))
            fila["postwarp_score"] = float(postwarp_data.get("postwarp_score", 0.0))
            fila["postwarp_retry_recommended"] = bool(postwarp_data.get("retry_recommended", False))

            if corners is None or len(corners) != 4:
                fila["pipeline_stage"] = "detection_failed"
                fila["error"] = "No se detectaron 4 esquinas validas"
                fila["analysis_status"] = "invalid_capture"
                fila["invalid_reasons"] = "invalid_detection"
                fila["usable_for_condition_model"] = False
                datos_cartas.append(fila)
                print("  -> Fallo: No se detectaron 4 esquinas validas.")
                continue

            if not fila["warp_success"]:
                fila["pipeline_stage"] = "warp_failed"
                fila["error"] = "No se pudo rectificar la carta"
                fila["analysis_status"] = "invalid_capture"
                fila["invalid_reasons"] = "warp_failed"
                fila["usable_for_condition_model"] = False
                datos_cartas.append(fila)
                print("  -> Fallo: No se pudo rectificar.")
                continue

            if not features_block.get("computed", False):
                fila["pipeline_stage"] = "features_failed"
                fila["error"] = features_block.get("reason", "No se pudieron calcular features")
                fila["analysis_status"] = "invalid_capture"
                fila["invalid_reasons"] = "features_not_available"
                fila["usable_for_condition_model"] = False
                datos_cartas.append(fila)
                print("  -> Fallo: No se pudieron calcular features.")
                continue

            estado = evaluar_estado_analisis(
                used_fallback=used_fallback,
                detection_metrics=detection_metrics,
                postwarp=postwarp_data,
                visuales=visuales,
                geometry=geometry,
                bordes=bordes,
                esquinas=esquinas,
            )
            fila.update(estado)

            fila["procesado_exito"] = True
            fila["pipeline_stage"] = "completed"
            print(f"  -> Exito. status={fila['analysis_status']}")

        except Exception as e:
            fila["pipeline_stage"] = "exception"
            fila["error"] = str(e)
            fila["analysis_status"] = "invalid_capture"
            fila["invalid_reasons"] = "exception"
            fila["usable_for_condition_model"] = False
            print(f"  -> Error procesando: {e}")

        datos_cartas.append(fila)

    if not datos_cartas:
        print("\nNo se logro extraer datos de ninguna imagen.")
        return

    df = pd.DataFrame(datos_cartas)

    columnas_inicio = [
        "image_id",
        "image_filename",
        "relative_path_from_raw",
        "categoria_carpeta_raw",
        "label_condition",
        "target_damaged",
        "analysis_status",
        "invalid_reasons",
        "usable_for_condition_model",
        "procesado_exito",
        "pipeline_stage",
        "error",
        "image_height",
        "image_width",
        "run_timestamp",
        "feature_schema_version",
    ]

    columnas_existentes_inicio = [c for c in columnas_inicio if c in df.columns]
    otras_columnas = [c for c in df.columns if c not in columnas_existentes_inicio]
    df = df[columnas_existentes_inicio + otras_columnas]

    salida_base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = salida_base.with_suffix(".csv")
    parquet_path = salida_base.with_suffix(".parquet")
    xlsx_path = salida_base.with_suffix(".xlsx")

    df.to_csv(csv_path, index=False)

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        print(f"No se pudo guardar parquet: {e}")

    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        print(f"No se pudo guardar Excel: {e}")

    print("\nDataset generado con exito!")
    print(f"-> Filas: {len(df)}")
    print(f"-> Columnas: {len(df.columns)}")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Excel: {xlsx_path}")

    if "label_condition" in df.columns:
        print("\nDistribucion por clase:")
        print(df["label_condition"].value_counts(dropna=False))

    if "analysis_status" in df.columns:
        print("\nDistribucion por analysis_status:")
        print(df["analysis_status"].value_counts(dropna=False))

    if "usable_for_condition_model" in df.columns:
        print("\nUsables para modelo de condicion:")
        print(df["usable_for_condition_model"].value_counts(dropna=False))


if __name__ == "__main__":
    CARPETA_ENTRADA = r"E:\Personal\Diplomado\Clases\MODULO 5\Proyecto Final\Gradix\data\raw"
    ARCHIVO_SALIDA_BASE = r"E:\Personal\Diplomado\Clases\MODULO 5\Proyecto Final\Gradix\data\processed\dataset_gradix"

    os.makedirs(CARPETA_ENTRADA, exist_ok=True)
    os.makedirs(os.path.dirname(ARCHIVO_SALIDA_BASE), exist_ok=True)

    procesar_lote_imagenes(CARPETA_ENTRADA, ARCHIVO_SALIDA_BASE)
