from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import numpy as np

from src.pipeline.card_analysis import analyze_card_image
from src.services.condition_model import predict_condition
from src.services.currency_converter import convert_eur_reference_amount
from src.services.openai_card_identifier import identify_card_from_warped_image
from src.ui.capture_guide import render_capture_guide
from src.ui.layout import render_header
from src.utils.logger import get_logger
from src.vision.image_loader import load_pil_image, pil_to_numpy, rgb_to_bgr, bgr_to_rgb
from src.vision.preprocess import resize_image
from src.services.pokemon_detector import get_pokemon_detector

logger = get_logger(__name__)


def render_pokemon_info_section(card_info: dict) -> None:
    """Renderiza la sección de información del Pokémon"""
    st.markdown("### 📊 Información del Pokémon Detectado")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre", card_info["nombre"])
        st.metric("Tipo", card_info["tipo_pokemon"] or "Desconocido")
    
    with col2:
        st.metric("Set", card_info["set"])
        st.metric("Número", card_info["numero"])
    
    with col3:
        st.metric("Rareza", card_info["rareza"] or "Sin rareza")
        if card_info["hp"]:
            st.metric("HP", card_info["hp"])
    
    # Imagen de la carta
    if card_info["imagen_url"]:
        st.image(card_info["imagen_url"], width=300, caption=f"{card_info['nombre']} - {card_info['set']}")
    
    # Ilustrador
    if card_info["ilustrador"]:
        st.caption(f"Ilustrador: {card_info['ilustrador']}")
    
    # Descripción
    if card_info["descripcion"]:
        st.markdown("**Descripción:**")
        st.write(card_info["descripcion"])
    
    # Habilidades
    if card_info["habilidades"]:
        st.markdown("**Habilidades:**")
        for ability in card_info["habilidades"]:
            st.write(f"- **{ability.get('name', 'Sin nombre')}**: {ability.get('effect', '')}")
    
    # Ataques
    if card_info["ataques"]:
        st.markdown("**Ataques:**")
        for attack in card_info["ataques"]:
            cost = attack.get("cost", [])
            damage = attack.get("damage", "?")
            st.write(f"- **{attack.get('name', 'Sin nombre')}** ({', '.join(cost)}) - {damage} daño")
            if attack.get("effect"):
                st.write(f"  - {attack['effect']}")


def render_capture_feedback(assessment: dict) -> None:
    level = assessment["capture_assessment"]
    message = assessment["capture_message"]

    if level == "buena":
        st.success(f"Captura adecuada: {message}")
    elif level == "mejorable":
        st.warning(f"Captura mejorable: {message}")
    else:
        st.error(f"Captura deficiente: {message}")


def search_and_render_pokemon_info(card_name: str) -> None:
    detector = get_pokemon_detector()
    results = detector.search_card_by_name(card_name)

    if not results:
        st.warning(f"No se encontró información en la API para: {card_name}")
        return

    selected_card = results[0]
    if selected_card.name and selected_card.name.lower() != card_name.lower():
        st.caption(f"Mejor coincidencia API: {selected_card.name}")
    detailed_card = detector.get_card_by_id(selected_card.id) if selected_card.id else None
    card_info = detector.format_card_info(detailed_card or selected_card)
    render_pokemon_info_section(card_info)


def main() -> None:
    render_header()
    render_capture_guide()

    st.markdown("### Carga de imagen")
    uploaded_file = st.file_uploader(
        "Sube una imagen de una carta Pokémon",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        st.info("Sube una imagen para comenzar.")
        return

    try:
        pil_image = load_pil_image(uploaded_file)
        if pil_image is None:
            st.error("No se pudo cargar la imagen.")
            return

        image_np = pil_to_numpy(pil_image)  
        image_bgr = rgb_to_bgr(image_np)    
        image_resized = resize_image(image_bgr)  

        # --- 1. PROCESAMIENTO VISUAL DE LA CARTA ---
        analysis_result = analyze_card_image(image_resized)

        detection_result = analysis_result["detection"]
        warp_block = analysis_result["warp"]
        postwarp_block = analysis_result["postwarp_validation"]
        features_block = analysis_result["features"]
        scores_block = analysis_result["scores"]
        model_result = predict_condition(features_block, scores_block)
        assessment_block = analysis_result["assessment"]
        debug_images = analysis_result["debug_images"]

        contour = detection_result.get("contour")
        corners = detection_result.get("corners")
        used_fallback = detection_result.get("used_fallback", True)
        detection_metrics = detection_result.get("metrics", {})

        warp_result = warp_block.get("data") if warp_block.get("computed") else None
        postwarp_result = (
            postwarp_block.get("data") if postwarp_block.get("computed") else None
        )
        
        warped_card = debug_images.get("warped_card")

        # --- 2. IDENTIFICACION DE CARTA (OPENAI -> BUSQUEDA) ---
        detector = get_pokemon_detector()
        openai_result = {"available": False, "reason": "Identificacion OpenAI no ejecutada."}
        identification_source = None
        identification_query = ""
        card_info = None

        def _is_usable_card_name(candidate: str) -> bool:
            candidate = (candidate or "").strip()
            if len(candidate) < 3:
                return False
            letters = sum(char.isalpha() for char in candidate)
            digits = sum(char.isdigit() for char in candidate)
            if letters < 3:
                return False
            if digits > max(2, letters):
                return False
            lowered = candidate.lower()
            if lowered in {"pokemon", "pokémon", "basic", "stage", "fase", "hp", "gx", "ex", "v"}:
                return False
            return True

        def _resolve_card_info(query_name: str, set_name: str = "", card_number: str = ""):
            results = detector.search_card_by_name(query_name)
            if not results:
                return None

            normalized_query = (query_name or "").strip().lower()
            normalized_set = (set_name or "").strip().lower()
            normalized_number = (card_number or "").strip().lower()

            def _candidate_score(card):
                score = 0
                candidate_name = (card.name or "").strip().lower()
                candidate_set = (card.set_name or "").strip().lower()
                candidate_number = (card.card_number or "").strip().lower()
                if candidate_name == normalized_query:
                    score += 10
                elif normalized_query and normalized_query in candidate_name:
                    score += 5
                if normalized_set:
                    if candidate_set == normalized_set:
                        score += 6
                    elif normalized_set in candidate_set:
                        score += 3
                if normalized_number:
                    if candidate_number == normalized_number:
                        score += 6
                    elif normalized_number in candidate_number or candidate_number in normalized_number:
                        score += 3
                return score

            selected_card = max(results, key=_candidate_score)
            detailed_card = detector.get_card_by_id(selected_card.id) if selected_card.id else None
            return detector.format_card_info(detailed_card or selected_card)

        if warped_card is not None:
            openai_result = identify_card_from_warped_image(warped_card)
            openai_name = (openai_result.get("name") or "").strip() if isinstance(openai_result, dict) else ""
            if openai_result.get("available") and _is_usable_card_name(openai_name):
                card_info = _resolve_card_info(
                    openai_name,
                    set_name=str(openai_result.get("set_name") or ""),
                    card_number=str(openai_result.get("card_number") or ""),
                )
                identification_source = "openai"
                identification_query = openai_name


        # --- 3. RENDERIZADO DE RESULTADOS DE VISIÓN ---
        edges = debug_images.get("canny_closed")
        if edges is None:
            edges = debug_images.get("canny")
        if edges is None:
            edges = debug_images.get("adaptive_closed")
        if edges is None:
            edges = debug_images.get("otsu_closed")

        contour_overlay = debug_images.get("detected_contour", image_resized)
        yolo_overlay = debug_images.get("yolo_bbox")
        coarse_quad_overlay = debug_images.get(
            "coarse_quad_overlay",
            debug_images.get("initial_quad_overlay"),
        )
        roi_quad_overlay = debug_images.get(
            "roi_quad_overlay",
            debug_images.get("refined_quad_overlay"),
        )
        roi_selected_quad_overlay = debug_images.get("roi_selected_quad_overlay")
        final_quad_overlay = debug_images.get(
            "final_quad_used_for_warp_overlay",
            debug_images.get("final_quad_overlay", contour_overlay),
        )
        coarse_quad_found = bool(detection_metrics.get("coarse_quad_found", 0) >= 0.5)
        roi_quad_found = bool(detection_metrics.get("roi_quad_found", 0) >= 0.5)
        final_quad_found = bool(detection_metrics.get("final_quad_found", 0) >= 0.5)
        final_quad_source = detection_metrics.get("final_quad_source", "none")
        final_quad_replaced_previous = bool(
            detection_metrics.get("final_quad_replaced_previous", 0) >= 0.5
        )
        final_quad_selection_reason = detection_metrics.get(
            "final_quad_selection_reason",
            "n/a",
        )
        refinement_roi = debug_images.get("refinement_roi")
        refinement_edges = debug_images.get("refinement_edges")
        structural_mask = debug_images.get("structural_mask")
        binary_view = debug_images.get("binary_view")
        hsv_view = debug_images.get("hsv_view")
        lab_view = debug_images.get("lab_view")
        multilayer_overlay = debug_images.get("multilayer_overlay")
        candidate_border_support_overlay = debug_images.get("candidate_border_support_overlay")
        refined_border_support_overlay = debug_images.get("refined_border_support_overlay")
        refinement_structural_mask = debug_images.get("refinement_structural_mask")
        binary_direct_candidate_mask = debug_images.get("binary_direct_candidate_mask")
        binary_direct_component_overlay = debug_images.get("binary_direct_component_overlay")
        binary_direct_quad_overlay = debug_images.get("binary_direct_quad_overlay")
        binary_thin_candidate_mask = debug_images.get("binary_thin_candidate_mask")
        binary_thin_contour_overlay = debug_images.get("binary_thin_contour_overlay")
        binary_thin_raw_quad_overlay = debug_images.get("binary_thin_raw_quad_overlay")
        binary_thin_local_accept_overlay = debug_images.get("binary_thin_local_accept_overlay")
        binary_thin_global_candidate_overlay = debug_images.get("binary_thin_global_candidate_overlay")
        binary_thin_rank_rejected_overlay = debug_images.get("binary_thin_rank_rejected_overlay")
        binary_thin_selected_overlay = debug_images.get("binary_thin_selected_overlay")

        warped_with_bbox = debug_images.get("warped_with_bbox")
        edge_overlay = debug_images.get("edge_overlay")
        corner_overlay = debug_images.get("corner_overlay")
        perspective_metrics = warp_result.get("metrics", {}) if warp_result else None

        visual_features = features_block.get("visual")
        geometry_features = features_block.get("geometry")
        centering_features = features_block.get("centering")
        edge_features = features_block.get("edge")
        corner_features = features_block.get("corner")
        whitening_surface_features = features_block.get("whitening_surface")

        quality_score = scores_block.get("capture_quality")
        gradix_score = scores_block.get("preliminary")
        centering_score = scores_block.get("centering")
        edge_score_result = scores_block.get("edge")
        corner_score_result = scores_block.get("corner")
        whitening_surface_score_result = scores_block.get("whitening_surface")
        condition_stub = scores_block.get("condition_stub_v1")
        condition_stub_v2 = scores_block.get("condition_stub_v2")
        condition_stub_v3 = scores_block.get("condition_stub_v3")
        condition_stub_v4 = scores_block.get("condition_stub_v4")
        capture_assessment = assessment_block.get("capture_quality")

        card_info = locals().get("card_info")
        postwarp_result = locals().get("postwarp_result")
        initial_quad_overlay = debug_images.get("initial_quad_overlay", coarse_quad_overlay)
        refined_quad_overlay = debug_images.get("refined_quad_overlay", roi_quad_overlay)

        def _read_value(obj, key, default=None):
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def _pick_first(obj, keys, default=None):
            for key in keys:
                value = _read_value(obj, key)
                if value not in (None, "", [], {}):
                    return value
            return default

        def _is_number(value):
            return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)

        def _fmt_score(value):
            return f"{float(value):.2f} / 10" if _is_number(value) else "Sin score"

        def _fmt_metric(value, decimals=3):
            if isinstance(value, bool):
                return "Si" if value else "No"
            if _is_number(value):
                return f"{float(value):.{decimals}f}"
            if value in (None, "", [], {}):
                return "N/A"
            return str(value)

        def _fmt_money(value, currency_code, decimals=2):
            if not _is_number(value):
                return "N/A"
            return f"{float(value):.{decimals}f} {currency_code}"

        def _score_label(score):
            if not _is_number(score):
                return "Sin evaluacion"
            if score >= 9.0:
                return "Excelente"
            if score >= 8.0:
                return "Muy buena"
            if score >= 7.0:
                return "Buena"
            if score >= 5.5:
                return "Mejorable"
            return "No confiable"

        def _render_optional_image(title, image, *, use_bgr=False, clamp=False, caption=None):
            if image is None:
                return False
            st.markdown(f"**{title}**")
            if use_bgr:
                st.image(bgr_to_rgb(image), width="stretch")
            else:
                st.image(image, width="stretch", clamp=clamp)
            if caption:
                st.caption(caption)
            return True

        def _collect_present_images(image_specs):
            return [spec for spec in image_specs if spec.get("image") is not None]

        def _render_image_grid(image_specs, columns=2):
            present_images = _collect_present_images(image_specs)
            if not present_images:
                return False
            for start in range(0, len(present_images), columns):
                row_specs = present_images[start : start + columns]
                row_columns = st.columns(columns)
                for column, spec in zip(row_columns, row_specs):
                    with column:
                        _render_optional_image(
                            spec["title"],
                            spec["image"],
                            use_bgr=spec.get("use_bgr", False),
                            clamp=spec.get("clamp", False),
                            caption=spec.get("caption"),
                        )
            return True

        final_score = None
        for score_obj, score_key in [
            (condition_stub_v4, "gradix_condition_stub_v4"),
            (condition_stub_v3, "gradix_condition_stub_v3"),
            (condition_stub_v2, "gradix_condition_stub_v2"),
            (condition_stub, "gradix_condition_stub"),
            (gradix_score, "gradix_preliminary_score"),
        ]:
            candidate = _read_value(score_obj, score_key)
            if _is_number(candidate):
                final_score = float(candidate)
                break

        card_name = _pick_first(card_info, ["nombre", "name", "card_name"]) or _read_value(openai_result, "name")
        card_set = _pick_first(card_info, ["set", "set_name"]) or _read_value(openai_result, "set_name")
        card_number = _pick_first(card_info, ["numero", "number", "collector_number"]) or _read_value(openai_result, "card_number")
        card_rarity = _pick_first(card_info, ["rareza", "rarity"]) or _read_value(openai_result, "rarity")
        card_type = _pick_first(card_info, ["tipo_pokemon", "type", "types"])
        card_hp = _pick_first(card_info, ["hp", "HP"]) or _read_value(openai_result, "hp")
        if isinstance(card_type, list):
            card_type = ", ".join(str(item) for item in card_type if item)

        findings = []
        capture_level = _read_value(capture_assessment, "capture_assessment")
        capture_message = _read_value(capture_assessment, "capture_message")
        if capture_level or capture_message:
            findings.append(
                f"Captura {str(capture_level or 'evaluada').capitalize()}: {capture_message or 'sin observaciones principales.'}"
            )
        for value, positive_text, negative_text in [
            (_read_value(centering_score, "centering_score"), "El centrado es un punto fuerte.", "El centrado se ve mejorable."),
            (_read_value(edge_score_result, "gradix_edge_score"), "Los bordes se ven consistentes.", "Los bordes merecen revision adicional."),
            (_read_value(corner_score_result, "gradix_corner_score"), "Las esquinas se conservan bien.", "Las esquinas concentran parte del deterioro visible."),
            (_read_value(whitening_surface_score_result, "gradix_whitening_surface_score"), "La superficie luce uniforme.", "La superficie presenta senales de desgaste."),
        ]:
            if not _is_number(value):
                continue
            if value >= 8.0:
                findings.append(positive_text)
            elif value < 6.0:
                findings.append(negative_text)
        findings = findings[:4]

        tab_resumen, tab_carta, tab_captura, tab_debug = st.tabs(
            ["Resumen", "Carta detectada", "Captura", "Debug técnico"]
        )

        with tab_resumen:
            info_col, image_col = st.columns([1.1, 1.4])
            with info_col:
                st.metric("Calificacion general", _fmt_score(final_score))
                st.caption(_score_label(final_score))
                if card_name:
                    st.subheader(card_name)

                summary_items = [
                    ("Set", card_set),
                    ("Numero", card_number),
                    ("Rareza", card_rarity),
                    ("Tipo", card_type),
                    ("HP", card_hp),
                ]
                summary_items = [(label, value) for label, value in summary_items if value not in (None, "", [], {})]
                if summary_items:
                    meta_columns = st.columns(min(len(summary_items), 5))
                    for column, (label, value) in zip(meta_columns, summary_items):
                        with column:
                            st.metric(label, str(value))

                if findings:
                    st.markdown("**Hallazgos principales**")
                    for finding in findings:
                        st.write(f"- {finding}")

                component_scores = [
                    ("Centrado", _read_value(centering_score, "centering_score")),
                    ("Bordes", _read_value(edge_score_result, "gradix_edge_score")),
                    ("Esquinas", _read_value(corner_score_result, "gradix_corner_score")),
                    ("Superficie", _read_value(whitening_surface_score_result, "gradix_whitening_surface_score")),
                ]
                component_scores = [(label, value) for label, value in component_scores if _is_number(value)]
                if component_scores:
                    component_columns = st.columns(len(component_scores))
                    for column, (label, value) in zip(component_columns, component_scores):
                        with column:
                            st.metric(label, _fmt_score(value))

                st.markdown("**Prediccion del modelo**")
                if _read_value(model_result, "available", False):
                    model_label = (
                        "Dañada"
                        if _read_value(model_result, "prediction") == 1
                        else "No dañada"
                    )
                    model_metrics = [("Predicción del modelo", model_label)]
                    probability_damaged = _read_value(model_result, "probability_damaged")
                    confidence = _read_value(model_result, "confidence")
                    if _is_number(probability_damaged):
                        model_metrics.append(
                            ("Probabilidad de daño", f"{float(probability_damaged) * 100:.1f}%")
                        )
                    if _is_number(confidence):
                        model_metrics.append(
                            ("Confianza", f"{float(confidence) * 100:.1f}%")
                        )
                    model_columns = st.columns(len(model_metrics))
                    for column, (label, value) in zip(model_columns, model_metrics):
                        with column:
                            st.metric(label, value)
                else:
                    st.info("Modelo supervisado no disponible para esta ejecución.")

            with image_col:
                _render_image_grid(
                    [
                        {"title": "Imagen original", "image": pil_image},
                        {"title": "Carta rectificada", "image": warped_card, "use_bgr": True},
                    ],
                    columns=2,
                )

        with tab_carta:
            if card_name:
                st.subheader(card_name)
            if identification_source:
                st.caption(f"Fuente de identificacion: {identification_source}")
            if identification_query:
                st.caption(f"Consulta usada: {identification_query}")
            if _is_number(_read_value(openai_result, "confidence")):
                st.caption(f"Confianza OpenAI: {float(_read_value(openai_result, 'confidence')) * 100:.1f}%")
            elif warped_card is not None and not _read_value(openai_result, "available", False):
                st.info("No se pudo identificar la carta con OpenAI en esta ejecucion.")

            card_items = [
                ("Set", card_set),
                ("Numero", card_number),
                ("Rareza", card_rarity),
                ("Tipo", card_type),
                ("HP", card_hp),
            ]
            card_items = [(label, value) for label, value in card_items if value not in (None, "", [], {})]
            if card_items:
                card_columns = st.columns(min(len(card_items), 5))
                for column, (label, value) in zip(card_columns, card_items):
                    with column:
                        st.metric(label, str(value))

            attacks = _read_value(card_info, "ataques") or _read_value(card_info, "attacks") or []
            if attacks:
                st.markdown("**Ataques detectados**")
                for attack in attacks:
                    if isinstance(attack, dict):
                        attack_name = attack.get("nombre") or attack.get("name") or attack.get("attack_name") or "Ataque"
                        attack_damage = attack.get("danio") or attack.get("damage")
                        attack_text = attack.get("descripcion") or attack.get("text") or attack.get("effect")
                        header = attack_name if attack_damage in (None, "") else f"{attack_name} | {attack_damage}"
                        st.write(f"- {header}")
                        if attack_text:
                            st.caption(str(attack_text))
                    else:
                        st.write(f"- {attack}")

            def _condition_category_from_score(score):
                if not _is_number(score):
                    return None
                score = float(score)
                if score >= 9.5:
                    return "mint"
                if score >= 8.5:
                    return "near_mint"
                if score >= 7.0:
                    return "lightly_played"
                if score >= 5.0:
                    return "moderately_played"
                if score >= 3.0:
                    return "heavily_played"
                return "damaged"

            def _condition_multiplier(category):
                return {
                    "mint": 1.00,
                    "near_mint": 0.95,
                    "lightly_played": 0.85,
                    "moderately_played": 0.70,
                    "heavily_played": 0.50,
                    "damaged": 0.30,
                }.get(category)

            def _condition_label(category):
                return {
                    "mint": "Mint",
                    "near_mint": "Near Mint",
                    "lightly_played": "Lightly Played",
                    "moderately_played": "Moderately Played",
                    "heavily_played": "Heavily Played",
                    "damaged": "Damaged",
                }.get(category, "Sin categoria")

            def _first_numeric_value(obj, keys):
                for key in keys:
                    value = _read_value(obj, key)
                    if _is_number(value):
                        return float(value)
                return None

            def _condition_specific_price(payload, category):
                if not category:
                    return None, None, None
                for key in ("condition_prices", "pricing_by_condition", "prices_by_condition"):
                    block = _read_value(payload, key)
                    if not isinstance(block, dict):
                        continue
                    candidate = block.get(category)
                    if _is_number(candidate):
                        return float(candidate), block.get("currency"), f"{key}.{category}"
                    if isinstance(candidate, dict):
                        direct_price = _first_numeric_value(candidate, ["marketPrice", "price", "value", "amount"])
                        if direct_price is not None:
                            return direct_price, candidate.get("currency") or block.get("currency"), f"{key}.{category}"
                return None, None, None

            market_info = _read_value(card_info, "mercado") or {}
            cardmarket_info = _read_value(market_info, "cardmarket") or {}
            tcgplayer_info = _read_value(market_info, "tcgplayer") or {}

            reference_eur = _first_numeric_value(cardmarket_info, ["trend", "avg7", "avg30", "low"])
            conversion_result = (
                convert_eur_reference_amount(reference_eur)
                if reference_eur is not None
                else {"available": False, "conversions": {}}
            )

            condition_category = _condition_category_from_score(final_score)
            condition_multiplier = _condition_multiplier(condition_category)
            api_condition_price, api_condition_currency, api_condition_source = _condition_specific_price(
                market_info,
                condition_category,
            )

            base_price_value = None
            base_price_currency = None
            base_price_source = None

            tcg_market_price = _first_numeric_value(tcgplayer_info, ["marketPrice", "midPrice", "lowPrice", "highPrice"])
            if tcg_market_price is not None:
                base_price_value = tcg_market_price
                base_price_currency = "USD"
                base_price_source = "tcgplayer"
            elif conversion_result.get("available") and _is_number((conversion_result.get("conversions") or {}).get("MXN")):
                base_price_value = float((conversion_result.get("conversions") or {}).get("MXN"))
                base_price_currency = "MXN"
                base_price_source = "cardmarket_eur_to_mxn"
            elif reference_eur is not None:
                base_price_value = float(reference_eur)
                base_price_currency = "EUR"
                base_price_source = "cardmarket"

            if api_condition_price is not None:
                precio_base_mercado = base_price_value
                precio_base_currency = base_price_currency
                precio_venta_estimado = float(api_condition_price)
                precio_venta_currency = api_condition_currency or base_price_currency or "USD"
                multiplicador_aplicado = (
                    precio_venta_estimado / precio_base_mercado
                    if _is_number(precio_base_mercado) and float(precio_base_mercado) > 0
                    else None
                )
                pricing_strategy_label = f"API por condicion ({api_condition_source})"
            else:
                precio_base_mercado = base_price_value
                precio_base_currency = base_price_currency
                precio_venta_estimado = (
                    float(precio_base_mercado) * float(condition_multiplier)
                    if _is_number(precio_base_mercado) and _is_number(condition_multiplier)
                    else None
                )
                precio_venta_currency = precio_base_currency
                multiplicador_aplicado = condition_multiplier
                pricing_strategy_label = "Multiplicador por condicion"

            precio_compra_estimado = (
                float(precio_venta_estimado) * 0.75
                if _is_number(precio_venta_estimado)
                else None
            )

            if cardmarket_info or tcgplayer_info:
                st.markdown("**Mercado**")
                market_col1, market_col2 = st.columns(2)
                with market_col1:
                    st.markdown("**Cardmarket (EUR)**")
                    cm_items = [
                        ("Trend", cardmarket_info.get("trend")),
                        ("Avg7", cardmarket_info.get("avg7")),
                        ("Avg30", cardmarket_info.get("avg30")),
                        ("Low", cardmarket_info.get("low")),
                    ]
                    cm_items = [(label, value) for label, value in cm_items if value not in (None, "", [], {})]
                    if cm_items:
                        cm_columns = st.columns(len(cm_items))
                        for column, (label, value) in zip(cm_columns, cm_items):
                            with column:
                                st.metric(label, _fmt_metric(value, decimals=2))
                        if conversion_result.get("available"):
                            st.caption("Conversion aproximada desde un valor de referencia en EUR.")
                            conversion_items = list((conversion_result.get("conversions") or {}).items())
                            if conversion_items:
                                conversion_columns = st.columns(len(conversion_items))
                                for column, (currency_code, converted_value) in zip(conversion_columns, conversion_items):
                                    with column:
                                        st.metric(currency_code, _fmt_money(converted_value, currency_code))
                        else:
                            st.caption("Conversiones de moneda no disponibles en este momento.")
                with market_col2:
                    st.markdown("**TCGplayer (USD)**")
                    tcg_items = [
                        ("Market", tcgplayer_info.get("marketPrice")),
                        ("Low", tcgplayer_info.get("lowPrice")),
                        ("Mid", tcgplayer_info.get("midPrice")),
                        ("High", tcgplayer_info.get("highPrice")),
                    ]
                    tcg_items = [(label, value) for label, value in tcg_items if value not in (None, "", [], {})]
                    if tcg_items:
                        tcg_columns = st.columns(len(tcg_items))
                        for column, (label, value) in zip(tcg_columns, tcg_items):
                            with column:
                                st.metric(label, _fmt_metric(value, decimals=2))

                st.markdown("**Ajuste por condicion**")
                pricing_items = [
                    ("Precio base de mercado", _fmt_money(precio_base_mercado, precio_base_currency) if _is_number(precio_base_mercado) and precio_base_currency else "N/A"),
                    ("Condicion estimada", _condition_label(condition_category)),
                    ("Multiplicador aplicado", f"{float(multiplicador_aplicado):.2f}x" if _is_number(multiplicador_aplicado) else "API"),
                    ("Precio de venta estimado", _fmt_money(precio_venta_estimado, precio_venta_currency) if _is_number(precio_venta_estimado) and precio_venta_currency else "N/A"),
                    ("Precio de compra estimado", _fmt_money(precio_compra_estimado, precio_venta_currency) if _is_number(precio_compra_estimado) and precio_venta_currency else "N/A"),
                ]
                pricing_columns = st.columns(len(pricing_items))
                for column, (label, value) in zip(pricing_columns, pricing_items):
                    with column:
                        st.metric(label, value)

                strategy_parts = []
                if base_price_source:
                    strategy_parts.append(f"base: {base_price_source}")
                if pricing_strategy_label:
                    strategy_parts.append(f"ajuste: {pricing_strategy_label}")
                if strategy_parts:
                    st.caption(" | ".join(strategy_parts))
            elif card_info:
                st.info("No hay datos de mercado disponibles para esta carta.")

        with tab_captura:
            if capture_assessment is not None:
                render_capture_feedback(capture_assessment)

            capture_items = [
                ("Cobertura", _read_value(capture_assessment, "coverage_ratio"), 3),
                ("Aspecto valido", _read_value(capture_assessment, "aspect_ok"), 3),
                ("Fallback usado", _read_value(capture_assessment, "used_fallback"), 3),
                ("Score preliminar", _read_value(gradix_score, "gradix_preliminary_score"), 2),
                ("Score de captura", _read_value(quality_score, "capture_quality_score"), 2),
                ("Cobertura geometrica", _read_value(geometry_features, "coverage_ratio"), 3),
                ("Post-warp valido", _read_value(postwarp_result, "postwarp_valid"), 3),
            ]
            capture_items = [(label, value, decimals) for label, value, decimals in capture_items if value not in (None, "", [], {})]
            for start in range(0, len(capture_items), 4):
                row = capture_items[start : start + 4]
                row_columns = st.columns(len(row))
                for column, (label, value, decimals) in zip(row_columns, row):
                    with column:
                        st.metric(label, _fmt_metric(value, decimals=decimals))

        with tab_debug:
            visual_debug = [
                {"title": "Imagen original", "image": pil_image},
                {"title": "Carta rectificada", "image": warped_card, "use_bgr": True},
                {"title": "Contorno detectado", "image": contour_overlay, "use_bgr": True},
                {"title": "YOLO - caja propuesta", "image": yolo_overlay, "use_bgr": True},
                {"title": "Bordes detectados", "image": edges, "clamp": True},
                {"title": "Mascara estructural", "image": structural_mask, "clamp": True},
                {"title": "Vista binaria", "image": binary_view, "clamp": True},
                {"title": "Overlay multicapa", "image": multilayer_overlay, "use_bgr": True},
                {"title": "Caja de contenido estimada", "image": warped_with_bbox, "use_bgr": True},
            ]
            if _collect_present_images(visual_debug):
                with st.expander("Visualizaciones principales", expanded=False):
                    _render_image_grid(visual_debug, columns=2)

            geometry_debug = [
                {"title": "Initial quad", "image": initial_quad_overlay, "use_bgr": True},
                {"title": "Refined quad", "image": refined_quad_overlay, "use_bgr": True},
                {"title": "Quad final", "image": final_quad_overlay, "use_bgr": True},
                {"title": "ROI refinada", "image": refinement_roi, "use_bgr": True},
                {"title": "Edges de refinamiento", "image": refinement_edges, "clamp": True},
                {"title": "Soporte candidato", "image": candidate_border_support_overlay, "use_bgr": True},
                {"title": "Soporte refinado", "image": refined_border_support_overlay, "use_bgr": True},
                {"title": "Mascara refinada", "image": refinement_structural_mask, "clamp": True},
            ]
            if _collect_present_images(geometry_debug):
                with st.expander("Geometria y refinamiento", expanded=False):
                    _render_image_grid(geometry_debug, columns=2)
                    trace_columns = st.columns(4)
                    for column, (label, value) in zip(
                        trace_columns,
                        [("Coarse", coarse_quad_found), ("ROI", roi_quad_found), ("Final", final_quad_found), ("Fallback", used_fallback)],
                    ):
                        with column:
                            st.metric(label, _fmt_metric(value))
                    if final_quad_source not in (None, "", "none"):
                        st.caption(f"Fuente final: {final_quad_source}")
                    if final_quad_selection_reason not in (None, "", "n/a"):
                        st.caption(f"Razon de seleccion: {final_quad_selection_reason}")

            technical_scores = [
                ("Capture Score", _read_value(quality_score, "capture_quality_score"), 2),
                ("Preliminary Score", _read_value(gradix_score, "gradix_preliminary_score"), 2),
                ("Centering Score", _read_value(centering_score, "centering_score"), 2),
                ("Edge Score", _read_value(edge_score_result, "gradix_edge_score"), 2),
                ("Corner Score", _read_value(corner_score_result, "gradix_corner_score"), 2),
                ("Surface Score", _read_value(whitening_surface_score_result, "gradix_whitening_surface_score"), 2),
                ("Blur", _read_value(visual_features, "blur_score"), 2),
                ("Brightness", _read_value(visual_features, "brightness_score"), 2),
                ("Contrast", _read_value(visual_features, "contrast_score"), 2),
            ]
            technical_scores = [(label, value, decimals) for label, value, decimals in technical_scores if value not in (None, "", [], {})]
            if technical_scores or edge_overlay is not None or corner_overlay is not None:
                with st.expander("Scores tecnicos", expanded=False):
                    for start in range(0, len(technical_scores), 4):
                        row = technical_scores[start : start + 4]
                        row_columns = st.columns(len(row))
                        for column, (label, value, decimals) in zip(row_columns, row):
                            with column:
                                st.metric(label, _fmt_metric(value, decimals=decimals))
                    _render_image_grid(
                        [
                            {"title": "Bandas de bordes", "image": edge_overlay, "use_bgr": True},
                            {"title": "Parches de esquinas", "image": corner_overlay, "use_bgr": True},
                        ],
                        columns=2,
                    )

            internal_metrics = [
                ("Best Score", detection_metrics.get("best_score") if detection_metrics else None, 3),
                ("Candidates", detection_metrics.get("num_candidates") if detection_metrics else None, 0),
                ("Detection Conf.", detection_metrics.get("detection_confidence") if detection_metrics else None, 3),
                ("Weak detection", detection_metrics.get("weak_detection", 0) >= 0.5 if detection_metrics else None, 3),
                ("YOLO disponible", detection_metrics.get("yolo_available", 0) >= 0.5 if detection_metrics else None, 3),
                ("Hard Rules", detection_metrics.get("hard_rules_ok", 0) >= 0.5 if detection_metrics else None, 3),
                ("Postwarp valido", _read_value(postwarp_result, "postwarp_valid"), 3),
                ("Postwarp score", _read_value(postwarp_result, "postwarp_score"), 3),
                ("Retry recomendado", _read_value(postwarp_result, "retry_recommended"), 3),
                ("Output Width", perspective_metrics.get("output_width") if perspective_metrics else None, 0),
                ("Output Height", perspective_metrics.get("output_height") if perspective_metrics else None, 0),
                ("Warp Aspect", perspective_metrics.get("stabilized_aspect_ratio") if perspective_metrics else None, 3),
            ]
            internal_metrics = [(label, value, decimals) for label, value, decimals in internal_metrics if value not in (None, "", [], {})]
            if internal_metrics:
                with st.expander("Metricas internas", expanded=False):
                    for start in range(0, len(internal_metrics), 4):
                        row = internal_metrics[start : start + 4]
                        row_columns = st.columns(len(row))
                        for column, (label, value, decimals) in zip(row_columns, row):
                            with column:
                                st.metric(label, _fmt_metric(value, decimals=decimals))
                    if not _read_value(model_result, "available", False):
                        model_reason = _read_value(model_result, "reason")
                        model_dir = _read_value(model_result, "model_dir")
                        if model_reason:
                            st.caption("Diagnóstico del modelo")
                            st.code(f"reason: {model_reason}")
                        if model_dir:
                            st.code(f"model_dir: {model_dir}")

    except Exception as exc:
        logger.exception("Error al procesar la imagen")
        st.error(f"Ocurrió un error al procesar la imagen: {exc}")


if __name__ == "__main__":
    main()
