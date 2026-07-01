"""Dashboard Streamlit para asesores en tienda: recomendaciones NBA."""

import base64
import os
from typing import Any

import requests
import streamlit as st

API_URL = 'http://localhost:8000'
LOGO_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'logo_haceb.png')

VISIT_REASONS_LABELS = {
    'cotizacion': 'Cotizacion de producto',
    'pago_cuota_credito': 'Pago de cuota de credito',
    'servicio_tecnico': 'Servicio tecnico',
    'reclamo': 'Reclamo o postventa',
}

SEGMENT_PLAYBOOK = {
    'VIP Comprador': {
        'lectura': 'Cliente de alto valor. Historial de compras premium, sensibilidad a precio baja.',
        'estrategia': 'Ofrecer gama alta y experiencias exclusivas. No competir por precio.',
    },
    'Cliente Financiado': {
        'lectura': 'Cliente que compra con credito Haceb. Responde bien a planes de cuotas.',
        'estrategia': 'Destacar cuota mensual, no precio total. Explicar beneficios del credito.',
    },
    'Visitante Servicios': {
        'lectura': 'Cliente activo con equipos en uso. Momento clave para recompra o upgrade.',
        'estrategia': 'Preguntar por el equipo actual. Proponer renovacion o complemento.',
    },
    'Cliente Dormido': {
        'lectura': 'Sin compras recientes. Necesita un motivo para volver.',
        'estrategia': 'Aplicar incentivo o promocion vigente. Preguntar por su ultima compra.',
    },
    'Nuevo Potencial': {
        'lectura': 'Cliente en construccion. Poco historial disponible.',
        'estrategia': 'Escuchar antes de vender. Conocer su necesidad real primero.',
    },
}


def load_logo_base64() -> str:
    """Carga el logo como base64 para embeber en HTML.

    Returns:
        Data URI del logo o cadena vacia si no existe el archivo.
    """
    if not os.path.exists(LOGO_PATH):
        return ''
    with open(LOGO_PATH, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{data}'


def request_prediction(id_cliente: str, motivo_visita: str) -> dict[str, Any]:
    """Consulta la API de inferencia para un cliente.

    Args:
        id_cliente: Identificador del cliente.
        motivo_visita: Motivo de visita registrado por el asesor.

    Returns:
        Diccionario con la respuesta de la API.

    Raises:
        ValueError: Si la API responde con error.
    """
    response = requests.post(
        f'{API_URL}/predict',
        json={'id_cliente': id_cliente, 'motivo_visita': motivo_visita},
        timeout=10,
    )
    if response.status_code != 200:
        detail = response.json().get('detail', 'Error desconocido')
        raise ValueError(detail)
    return response.json()


def get_api_status() -> bool:
    """Verifica si la API esta disponible.

    Returns:
        Estado booleano de la API.
    """
    try:
        response = requests.get(f'{API_URL}/health', timeout=5)
    except requests.RequestException:
        return False
    return response.status_code == 200


def opportunity_message(nivel: str) -> tuple[str, str]:
    """Devuelve el titulo y el subtitulo de la oportunidad segun el nivel.

    Args:
        nivel: Nivel de propension (ALTO, MEDIO, BAJO).

    Returns:
        Tupla con titulo y subtitulo mostrados al asesor.
    """
    if nivel == 'ALTO':
        return 'Oportunidad de venta hoy', 'Cliente listo para cerrar. Ejecuta la accion recomendada con confianza.'
    if nivel == 'MEDIO':
        return 'Oportunidad con acompanamiento', 'Cliente interesado. Escucha primero y adapta la propuesta.'
    return 'Prioridad en atencion', 'Enfocate en atender su solicitud. Registra la visita para futuras oportunidades.'


st.set_page_config(
    page_title='Motor NBA | Haceb',
    page_icon='H',
    layout='wide',
    initial_sidebar_state='collapsed',
)

logo_b64 = load_logo_base64()


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --haceb-navy: #002D62;
    --haceb-navy-2: #003F87;
    --haceb-navy-3: #001A3D;
    --haceb-red: #E31E24;
    --haceb-red-2: #C9181D;
    --haceb-red-3: #A81317;
    --bg: #FFFFFF;
    --bg-subtle: #F7F9FC;
    --bg-navy-soft: #EEF3FA;
    --bg-red-soft: #FEF2F2;
    --border: #E4E9F0;
    --border-strong: #D0D8E3;
    --text: #0F172A;
    --text-2: #475569;
    --text-3: #64748B;
    --text-4: #94A3B8;
    --success: #16A34A;
    --success-soft: #ECFDF5;
    --warn: #D97706;
    --warn-soft: #FFFBEB;
    --shadow-sm: 0 1px 2px rgba(0, 45, 98, 0.05);
    --shadow-md: 0 4px 12px rgba(0, 45, 98, 0.08);
    --shadow-lg: 0 12px 32px rgba(0, 45, 98, 0.12);
}

html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg-subtle); color: var(--text); }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1180px !important;
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

/* Nav bar navy — Haceb identity */
.nav {
    background: linear-gradient(135deg, var(--haceb-navy) 0%, var(--haceb-navy-2) 100%);
    border-radius: 0 0 20px 20px;
    padding: 20px 32px;
    margin: 0 -1rem 0;
    display: flex; justify-content: space-between; align-items: center;
    position: relative;
    box-shadow: var(--shadow-md);
}
.nav::before {
    content: ''; position: absolute; top: 0; right: 0; bottom: 0; width: 45%;
    background: radial-gradient(circle at 80% 50%, rgba(227,30,36,0.18) 0%, transparent 60%);
    pointer-events: none; border-radius: 0 0 20px 0;
}
.nav::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, transparent 0%, var(--haceb-red) 30%, var(--haceb-red) 70%, transparent 100%);
    border-radius: 0 0 20px 20px;
}
.nav-left { display: flex; align-items: center; gap: 18px; position: relative; z-index: 1; }
.logo-chip {
    background: #fff; border-radius: 10px;
    padding: 8px 14px;
    display: flex; align-items: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.18);
}
.logo-chip img { height: 26px; display: block; }
.nav-divider {
    width: 1px; height: 28px;
    background: rgba(255,255,255,0.2);
}
.nav-brand-block { display: flex; flex-direction: column; gap: 2px; }
.nav-brand {
    font-size: 15px; font-weight: 700; color: #fff;
    letter-spacing: -0.2px; line-height: 1;
}
.nav-brand-sub {
    font-size: 11px; font-weight: 500;
    color: rgba(255,255,255,0.7);
    letter-spacing: 0.2px;
}
.nav-status {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 8px 14px; border-radius: 999px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.15);
    font-size: 12px; font-weight: 600; color: #fff;
    position: relative; z-index: 1;
    backdrop-filter: blur(8px);
}
.nav-status .dot {
    width: 7px; height: 7px; border-radius: 50%;
}
.nav-status.ok .dot { background: #4ADE80; box-shadow: 0 0 0 3px rgba(74,222,128,0.25); }
.nav-status.off .dot { background: #FCA5A5; box-shadow: 0 0 0 3px rgba(252,165,165,0.25); }

.hero {
    background: var(--bg);
    border-radius: 16px;
    padding: 40px 44px;
    margin: 32px 0 32px;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; right: -60px; top: -60px;
    width: 240px; height: 240px; border-radius: 50%;
    background: radial-gradient(circle, rgba(227,30,36,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: ''; position: absolute; left: 0; top: 40px; bottom: 40px; width: 4px;
    background: linear-gradient(180deg, var(--haceb-red) 0%, var(--haceb-navy) 100%);
    border-radius: 0 4px 4px 0;
}
.hero-eyebrow {
    font-size: 12px; font-weight: 700; color: var(--haceb-red);
    text-transform: uppercase; letter-spacing: 0.9px;
    margin-bottom: 14px;
    display: inline-flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
    content: ''; width: 22px; height: 2px; background: var(--haceb-red); border-radius: 2px;
}
.hero-title {
    font-size: 32px; font-weight: 800; color: var(--haceb-navy);
    letter-spacing: -1px; line-height: 1.15;
    margin: 0 0 12px; max-width: 700px;
}
.hero-title .accent {
    color: var(--haceb-red);
}
.hero-sub {
    font-size: 15px; color: var(--text-2);
    line-height: 1.6; max-width: 620px; margin: 0;
}

.section-label {
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 10px; font-weight: 800; color: #fff;
    text-transform: uppercase; letter-spacing: 0.9px;
    margin-bottom: 12px;
    padding: 5px 11px;
    background: var(--haceb-navy);
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0,45,98,0.2);
}
.section-label::before {
    content: ''; width: 6px; height: 6px; border-radius: 50%;
    background: var(--haceb-red);
}
.section-title {
    font-size: 22px; font-weight: 800; color: var(--haceb-navy);
    letter-spacing: -0.5px; margin: 0 0 6px;
}
.section-sub {
    font-size: 13px; color: var(--text-3);
    line-height: 1.55; margin: 0 0 22px; max-width: 460px;
}

/* Form container styled */
div[data-testid="stForm"] {
    border: 1px solid var(--border) !important;
    border-top: 4px solid var(--haceb-navy) !important;
    border-radius: 12px !important;
    padding: 26px !important;
    background: var(--bg) !important;
    box-shadow: var(--shadow-md) !important;
}
div[data-testid="stForm"] > div:first-child { gap: 18px !important; }

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    font-size: 12px !important; font-weight: 700 !important;
    color: var(--haceb-navy) !important; margin-bottom: 8px !important;
    text-transform: uppercase; letter-spacing: 0.4px;
}
div[data-testid="stTextInput"] input {
    border-radius: 8px !important;
    border: 1.5px solid var(--border-strong) !important;
    background: var(--bg) !important; padding: 12px 14px !important;
    font-size: 14px !important; color: var(--text) !important;
    box-shadow: none !important; transition: all 0.15s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--haceb-navy) !important;
    box-shadow: 0 0 0 3px rgba(0,45,98,0.15) !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border: 1.5px solid var(--border-strong) !important;
    background: var(--bg) !important; min-height: 44px !important;
    box-shadow: none !important;
}

/* Button — Haceb red, forced white text */
.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100% !important;
    border-radius: 8px !important;
    border: none !important;
    min-height: 48px !important;
    background: linear-gradient(135deg, var(--haceb-red) 0%, var(--haceb-red-2) 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
    transition: all 0.15s !important;
    box-shadow: 0 6px 16px rgba(227,30,36,0.28) !important;
}
.stButton > button *,
div[data-testid="stFormSubmitButton"] > button *,
.stButton > button p,
div[data-testid="stFormSubmitButton"] > button p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: linear-gradient(135deg, var(--haceb-red-2) 0%, var(--haceb-red-3) 100%) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 22px rgba(227,30,36,0.35) !important;
}
.stButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus {
    box-shadow: 0 0 0 4px rgba(227,30,36,0.25), 0 6px 16px rgba(227,30,36,0.28) !important;
    outline: none !important;
}

.hint {
    margin-top: 24px;
    padding: 18px 20px;
    background: linear-gradient(135deg, var(--haceb-navy) 0%, var(--haceb-navy-2) 100%);
    border-radius: 12px;
    color: #fff;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow-md);
}
.hint::after {
    content: ''; position: absolute; right: -40px; bottom: -40px;
    width: 140px; height: 140px; border-radius: 50%;
    background: radial-gradient(circle, rgba(227,30,36,0.2) 0%, transparent 70%);
    pointer-events: none;
}
.hint-title {
    font-size: 11px; font-weight: 700; color: #fff;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
    position: relative; z-index: 1;
}
.hint-title::before {
    content: ''; width: 14px; height: 2px; background: var(--haceb-red); border-radius: 2px;
}
.hint-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; font-size: 13px;
    position: relative; z-index: 1;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}
.hint-row:last-child { border-bottom: none; }
.hint-row .k { color: rgba(255,255,255,0.75); font-weight: 500; }
.hint-row .v {
    color: #fff; font-weight: 700; font-size: 12px;
    background: rgba(255,255,255,0.15);
    padding: 3px 10px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.15);
}

.empty {
    background: var(--bg);
    border: 2px dashed var(--border-strong);
    border-radius: 14px;
    padding: 72px 32px; text-align: center;
    position: relative; overflow: hidden;
}
.empty::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 160px; height: 160px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,45,98,0.05) 0%, transparent 70%);
}
.empty-icon {
    width: 56px; height: 56px; border-radius: 14px;
    background: linear-gradient(135deg, var(--haceb-navy) 0%, var(--haceb-navy-2) 100%);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 20px;
    color: #fff;
    font-size: 26px; font-weight: 800;
    box-shadow: 0 8px 20px rgba(0,45,98,0.25);
    position: relative; z-index: 1;
}
.empty-title {
    font-size: 16px; font-weight: 700; color: var(--haceb-navy); margin-bottom: 6px;
    position: relative; z-index: 1;
}
.empty-copy {
    font-size: 13px; color: var(--text-3);
    max-width: 340px; margin: 0 auto; line-height: 1.6;
    position: relative; z-index: 1;
}

.opp {
    padding: 28px 30px; border-radius: 14px;
    color: #fff; position: relative; overflow: hidden;
    margin-bottom: 24px;
    box-shadow: var(--shadow-lg);
}
.opp.alto {
    background: linear-gradient(135deg, var(--haceb-red) 0%, var(--haceb-red-2) 60%, var(--haceb-red-3) 100%);
}
.opp.medio {
    background: linear-gradient(135deg, #F59E0B 0%, #D97706 60%, #B45309 100%);
}
.opp.bajo {
    background: linear-gradient(135deg, var(--haceb-navy-2) 0%, var(--haceb-navy) 60%, var(--haceb-navy-3) 100%);
}
.opp::after {
    content: ''; position: absolute; right: -80px; top: -80px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, transparent 70%);
    pointer-events: none;
}
.opp-tag {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.22);
    font-size: 10px; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.9px;
    color: #fff; margin-bottom: 14px;
    position: relative; z-index: 1;
    border: 1px solid rgba(255,255,255,0.2);
}
.opp-tag .dot {
    width: 6px; height: 6px; border-radius: 50%; background: #fff;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.3);
}
.opp-title {
    font-size: 24px; font-weight: 800; letter-spacing: -0.6px;
    line-height: 1.2; margin: 0 0 8px;
    position: relative; z-index: 1;
}
.opp-sub {
    font-size: 14px; color: rgba(255,255,255,0.92);
    line-height: 1.55; margin: 0; max-width: 500px;
    position: relative; z-index: 1;
}
.opp-client {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 22px; padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.22);
    font-size: 12px;
    position: relative; z-index: 1;
}
.opp-client .k { color: rgba(255,255,255,0.75); font-weight: 500; }
.opp-client .v {
    color: #fff; font-weight: 700;
    background: rgba(255,255,255,0.18);
    padding: 5px 12px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.15);
}

.section-block { margin-bottom: 22px; }
.block-label {
    font-size: 11px; font-weight: 800; color: var(--haceb-navy);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 12px;
    display: inline-flex; align-items: center; gap: 8px;
}
.block-label::before {
    content: ''; width: 4px; height: 14px; background: var(--haceb-red); border-radius: 2px;
}

.offer-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 5px solid var(--haceb-navy);
    border-radius: 10px; padding: 22px 24px;
    box-shadow: var(--shadow-sm);
    position: relative; overflow: hidden;
}
.offer-card::after {
    content: ''; position: absolute; right: 20px; top: 50%; transform: translateY(-50%);
    width: 60px; height: 60px; border-radius: 50%;
    background: radial-gradient(circle, rgba(0,45,98,0.05) 0%, transparent 70%);
}
.offer-title {
    font-size: 22px; font-weight: 800; color: var(--haceb-navy);
    letter-spacing: -0.5px; line-height: 1.2;
    margin: 0 0 6px;
}
.offer-hist {
    font-size: 13px; color: var(--text-3); margin: 0;
}
.offer-hist strong { color: var(--haceb-navy); font-weight: 700; }

.script-card {
    background: linear-gradient(135deg, var(--haceb-navy) 0%, var(--haceb-navy-2) 100%);
    color: #fff;
    border-radius: 12px; padding: 26px 28px;
    position: relative; overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.script-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
    background: linear-gradient(180deg, var(--haceb-red) 0%, #F59E0B 100%);
}
.script-card::after {
    content: ''; position: absolute; right: -60px; bottom: -60px;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(227,30,36,0.22) 0%, transparent 70%);
    pointer-events: none;
}
.script-head {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px;
    font-weight: 800; color: rgba(255,255,255,0.65);
    margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
    position: relative; z-index: 1;
}
.script-head::before {
    content: ''; width: 16px; height: 2px; background: var(--haceb-red); border-radius: 2px;
}
.script-text {
    font-size: 16px; font-weight: 500; line-height: 1.6;
    color: #fff; letter-spacing: -0.1px;
    margin: 0;
    position: relative; z-index: 1;
}

.playbook {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 14px;
}
.pb-item {
    background: var(--bg);
    border: 1px solid var(--border);
    border-top: 3px solid var(--haceb-red);
    border-radius: 10px;
    padding: 18px 20px;
    transition: all 0.15s;
    box-shadow: var(--shadow-sm);
}
.pb-item:nth-child(2) {
    border-top-color: var(--haceb-navy);
}
.pb-item:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.pb-key {
    font-size: 11px; font-weight: 800; color: var(--haceb-navy);
    text-transform: uppercase; letter-spacing: 0.7px;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
}
.pb-value {
    font-size: 13px; color: var(--text);
    line-height: 1.6; font-weight: 500;
}

.footer {
    margin-top: 44px; padding-top: 20px;
    border-top: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; color: var(--text-4);
}
.footer .brand { color: var(--haceb-navy); font-weight: 700; }

div[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid rgba(227,30,36,0.25) !important;
    box-shadow: none !important;
    background: var(--bg-red-soft) !important;
    color: var(--haceb-red-2) !important;
}
</style>""", unsafe_allow_html=True)


api_ok = get_api_status()
status_class = 'ok' if api_ok else 'off'
status_text = 'Sistema activo' if api_ok else 'Sistema fuera de linea'

if logo_b64:
    logo_html = f'<img src="{logo_b64}" alt="Haceb">'
else:
    logo_html = '<span style="font-weight:800;font-size:16px;color:#002D62;">Haceb</span>'

st.markdown(
    f"""<div class="nav">
      <div class="nav-left">
        <div class="logo-chip">{logo_html}</div>
        <div class="nav-divider"></div>
        <div class="nav-brand-block">
          <div class="nav-brand">Asistente de Venta</div>
          <div class="nav-brand-sub">Motor NBA para piso de tienda</div>
        </div>
      </div>
      <div class="nav-status {status_class}">
        <span class="dot"></span>{status_text}
      </div>
    </div>""",
    unsafe_allow_html=True,
)


st.markdown(
    """<div class="hero">
      <div class="hero-eyebrow">Asistente para el asesor</div>
      <h1 class="hero-title">Sabe que ofrecerle a cada cliente que <span class="accent">entra a la tienda.</span></h1>
      <p class="hero-sub">Ingresa el ID del cliente y su motivo de visita. El asistente te dice si hay oportunidad, que producto ofrecerle y como iniciar la conversacion.</p>
    </div>""",
    unsafe_allow_html=True,
)


if 'prediction_data' not in st.session_state:
    st.session_state['prediction_data'] = None
if 'prediction_error' not in st.session_state:
    st.session_state['prediction_error'] = None


form_col, result_col = st.columns([1, 1.5], gap='large')

with form_col:
    st.markdown(
        """<div class="section-label"><span>Paso 1</span></div>
        <h2 class="section-title">Consulta el cliente</h2>
        <p class="section-sub">Ingresa sus datos y genera la recomendacion.</p>""",
        unsafe_allow_html=True,
    )

    with st.form('advisor_form', clear_on_submit=False):
        id_cliente = st.text_input(
            'ID del cliente',
            placeholder='cli_001',
        )
        motivo = st.selectbox(
            'Motivo de la visita',
            options=list(VISIT_REASONS_LABELS.keys()),
            format_func=lambda k: VISIT_REASONS_LABELS[k],
        )
        submitted = st.form_submit_button('Ver recomendacion')

    st.markdown(
        """<div class="hint">
          <div class="hint-title">IDs de prueba</div>
          <div class="hint-row"><span class="k">Cliente activo</span><span class="v">cli_001</span></div>
          <div class="hint-row"><span class="k">Cliente frecuente</span><span class="v">cli_043</span></div>
          <div class="hint-row"><span class="k">Cliente dormido</span><span class="v">cli_120</span></div>
        </div>""",
        unsafe_allow_html=True,
    )


with result_col:
    st.markdown(
        """<div class="section-label"><span>Paso 2</span></div>
        <h2 class="section-title">Tu jugada con este cliente</h2>
        <p class="section-sub">Que oportunidad hay, que ofrecerle y como abordarlo.</p>""",
        unsafe_allow_html=True,
    )

    if submitted:
        st.session_state['prediction_data'] = None
        st.session_state['prediction_error'] = None
        if not id_cliente:
            st.session_state['prediction_error'] = 'Ingresa un ID de cliente para consultar.'
        else:
            try:
                st.session_state['prediction_data'] = request_prediction(
                    id_cliente=id_cliente,
                    motivo_visita=motivo,
                )
            except ValueError as exc:
                st.session_state['prediction_error'] = str(exc)
            except requests.ConnectionError:
                st.session_state['prediction_error'] = 'No se pudo conectar con el sistema. Avisa al soporte de tienda.'
            except requests.RequestException:
                st.session_state['prediction_error'] = 'La consulta fallo. Intentalo de nuevo en unos segundos.'

    prediction_data = st.session_state['prediction_data']
    prediction_error = st.session_state['prediction_error']

    if prediction_error:
        st.error(prediction_error)
    elif prediction_data:
        nivel = prediction_data['nivel']
        nivel_cls = nivel.lower()
        motivo_label = VISIT_REASONS_LABELS.get(motivo, motivo)
        segmento = prediction_data['segmento']
        playbook = SEGMENT_PLAYBOOK.get(
            segmento,
            {'lectura': 'Perfil comercial asignado.', 'estrategia': 'Escucha y adapta la propuesta.'},
        )
        cat_next = prediction_data['next_best_category'].replace('_', ' ').title()
        cat_hist = prediction_data['categoria_favorita'].replace('_', ' ').title()
        opp_title, opp_sub = opportunity_message(nivel)

        st.markdown(
            f"""<div class="opp {nivel_cls}">
              <div class="opp-tag"><span class="dot"></span>Prioridad {nivel}</div>
              <div class="opp-title">{opp_title}</div>
              <p class="opp-sub">{opp_sub}</p>
              <div class="opp-client">
                <span class="k">Cliente {prediction_data['id_cliente']} &middot; Visita por {motivo_label.lower()}</span>
                <span class="v">{segmento}</span>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="section-block">
              <div class="block-label">Que ofrecerle</div>
              <div class="offer-card">
                <div class="offer-title">{cat_next}</div>
                <p class="offer-hist">Su categoria historica favorita ha sido <strong>{cat_hist}</strong>.</p>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="section-block">
              <div class="block-label">Que decirle</div>
              <div class="script-card">
                <div class="script-head">Guion sugerido</div>
                <p class="script-text">{prediction_data['accion_recomendada']}</p>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="section-block">
              <div class="block-label">Contexto del cliente</div>
              <div class="playbook">
                <div class="pb-item">
                  <div class="pb-key">Como es este cliente</div>
                  <div class="pb-value">{playbook['lectura']}</div>
                </div>
                <div class="pb-item">
                  <div class="pb-key">Como abordarlo</div>
                  <div class="pb-value">{playbook['estrategia']}</div>
                </div>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="empty">
              <div class="empty-icon">H</div>
              <div class="empty-title">Aun no hay recomendacion</div>
              <div class="empty-copy">Ingresa el ID del cliente y su motivo de visita para generar la jugada.</div>
            </div>""",
            unsafe_allow_html=True,
        )


st.markdown(
    """<div class="footer">
      <div>Motor de <span class="brand">Proxima Mejor Accion</span> &middot; Piso de tienda</div>
      <div>Haceb 2026</div>
    </div>""",
    unsafe_allow_html=True,
)
