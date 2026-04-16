import streamlit as st


def render_capture_guide() -> None:
    st.markdown("### Modo de captura asistida")

    st.info(
        "Coloca una sola carta dentro del encuadre, procura que esté de frente, "
        "con fondo simple, buena iluminación y sin reflejos fuertes."
    )

    guide_col1, guide_col2 = st.columns([1, 1])

    with guide_col1:
        st.markdown("#### Recomendaciones")
        st.markdown(
            """
            - Usa una sola carta por imagen.
            - Procura que la carta ocupe buena parte de la foto.
            - Evita inclinación excesiva.
            - Evita sombras y reflejos fuertes.
            - Usa fondo liso o con poco ruido visual.
            - Toma la foto lo más frontal posible.
            """
        )

    with guide_col2:
        st.markdown("#### Encuadre ideal")
        st.markdown(
            """
            ```text
            ┌───────────────────────────────┐
            │                               │
            │        ┌─────────────┐        │
            │        │             │        │
            │        │   CARTA     │        │
            │        │             │        │
            │        └─────────────┘        │
            │                               │
            └───────────────────────────────┘
            ```
            """
        )
        st.caption(
            "La carta debe verse completa, centrada y ocupando una parte importante del encuadre."
        )