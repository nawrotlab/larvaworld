from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import panel as pn
from bokeh.models import InlineStyleSheet

from larvaworld.portal.landing_registry import (
    ITEMS,
    LANES,
    QUICK_START_DEFAULT_MODE,
    QUICK_START_MODES,
)
from larvaworld.portal.panel_components import (
    PORTAL_RAW_CSS,
    build_footer,
    build_template_header,
    render_card,
    render_lane,
)
from larvaworld.portal.notebook_workspace import (
    notebook_names_by_item,
    notebook_urls_by_item,
)
from larvaworld.portal.registry_logic import validate_registry
from larvaworld.portal.workspace import get_active_workspace


def _load_banner_gif_data_uri(filename: str) -> str:
    gif_path = Path(__file__).with_name("icons") / "gifs" / filename
    try:
        encoded = base64.b64encode(gif_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:image/gif;base64,{encoded}"


def _banner_slides() -> list[dict[str, str]]:
    slide_configs = [
        {
            "filename": "Bisegmental_simplification_of_the_larva_body.gif",
            "title": "Bisegmental simplification of the larva body",
            "description": (
                "This simulation illustrates how a reduced body representation preserves key "
                "locomotion dynamics. It is useful for fast model experiments and sensitivity checks."
            ),
            "url": "https://computational-systems-neuroscience.de/wp-content/uploads/2024/10/1.mp4",
        },
        {
            "filename": "Locomotory_model_for_Drosophila_larva.gif",
            "title": "Locomotory model for Drosophila larva",
            "description": (
                "The locomotory model combines crawling rhythms and directional control in a "
                "single pipeline. Use it to inspect how motor components shape trajectory behavior."
            ),
            "url": "https://computational-systems-neuroscience.de/wp-content/uploads/2024/10/2.mp4",
        },
        {
            "filename": "Linear_Angular_speed_during_a_larva_trajectory.gif",
            "title": "Linear & angular speed during a larva trajectory",
            "description": (
                "Speed traces expose coupling between forward motion and turning phases. "
                "This view helps compare temporal kinematic signatures across conditions."
            ),
            "url": "https://computational-systems-neuroscience.de/wp-content/uploads/2024/10/3.mp4",
        },
        {
            "filename": "Dispersion_for_real_VS_simulated_larvae.gif",
            "title": "Dispersion for real VS simulated larvae",
            "description": (
                "Dispersion summaries benchmark simulated groups against observed cohort spread. "
                "Use this comparison to validate spatial exploration realism."
            ),
            "url": "https://computational-systems-neuroscience.de/wp-content/uploads/2024/10/5.mp4",
        },
        {
            "filename": "Growth_over_7_hours.gif",
            "title": "Growth over 7 hours",
            "description": (
                "Long-horizon growth dynamics reveal substrate-dependent developmental trends. "
                "This scenario links foraging context with life-history model outputs."
            ),
            "url": "https://computational-systems-neuroscience.de/wp-content/uploads/2025/07/growth_7hours.mp4",
        },
    ]

    slides: list[dict[str, str]] = []
    for config in slide_configs:
        data_uri = _load_banner_gif_data_uri(config["filename"])
        if not data_uri:
            continue
        slides.append(
            {
                "title": config["title"],
                "description": config["description"],
                "url": config["url"],
                "data_uri": data_uri,
            }
        )
    return slides


def _banner_media_html(slide: dict[str, str], *, play_token: int) -> str:
    return (
        '<img class="lw-portal-banner-gif" '
        f'src="{slide["data_uri"]}#play-{play_token}" alt="{escape(slide["title"])}" />'
    )


def _banner_text_html(slide: dict[str, str]) -> str:
    return (
        '<div class="lw-portal-banner-title">'
        f'{escape(slide["title"])}'
        "</div>"
        '<div class="lw-portal-banner-description">'
        f'{escape(slide["description"])}'
        "</div>"
        f'<a class="lw-portal-banner-link" href="{escape(slide["url"])}" '
        'target="_blank" rel="noopener noreferrer">Learn more</a>'
    )


def landing_app() -> pn.viewable.Viewable:
    pn.extension(raw_css=[PORTAL_RAW_CSS])
    if get_active_workspace() is None:
        return pn.Column(
            pn.pane.HTML(
                (
                    '<script>window.location.replace("/");</script>'
                    '<div style="max-width:720px;margin:36px auto;padding:16px 18px;'
                    "border:1px solid rgba(0,0,0,0.15);border-radius:12px;"
                    'font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">'
                    '<h3 style="margin:0 0 10px 0;">Workspace setup required</h3>'
                    '<p style="margin:0;">Redirecting to workspace setup...</p>'
                    "</div>"
                ),
                margin=0,
            ),
            sizing_mode="stretch_width",
        )
    validate_registry(strict=True)

    template = pn.template.MaterialTemplate(
        title="Larvaworld Portal",
        header_background="#f5a142",
        header_color="#111111",
    )
    root = pn.Column(css_classes=["lw-portal-root"], sizing_mode="stretch_width")

    topbar = build_template_header()
    template.header.append(topbar)
    slides = _banner_slides()
    if slides:
        active_slide = {"index": 0, "play_token": 0}
        media_pane = pn.pane.HTML(
            _banner_media_html(slides[0], play_token=0),
            margin=0,
            css_classes=["lw-portal-banner-media"],
            sizing_mode="stretch_width",
        )
        text_pane = pn.pane.HTML(
            _banner_text_html(slides[0]),
            margin=0,
            css_classes=["lw-portal-banner-copy"],
            sizing_mode="stretch_width",
        )
        banner_main = pn.Row(
            media_pane,
            text_pane,
            css_classes=["lw-portal-banner-main"],
            sizing_mode="stretch_width",
            margin=0,
        )
        prev_button = pn.widgets.Button(
            name="‹",
            button_type="default",
            width=30,
            height=30,
            margin=0,
            css_classes=["lw-portal-banner-nav-btn", "lw-portal-banner-nav-btn--left"],
        )
        next_button = pn.widgets.Button(
            name="›",
            button_type="default",
            width=30,
            height=30,
            margin=0,
            css_classes=["lw-portal-banner-nav-btn", "lw-portal-banner-nav-btn--next"],
        )

        def _set_slide(index: int) -> None:
            active_slide["index"] = index % len(slides)
            active_slide["play_token"] += 1
            current = slides[active_slide["index"]]
            media_pane.object = _banner_media_html(
                current, play_token=active_slide["play_token"]
            )
            text_pane.object = _banner_text_html(current)
            if active_slide["index"] % 2 == 0:
                banner_main.css_classes = ["lw-portal-banner-main"]
            else:
                banner_main.css_classes = [
                    "lw-portal-banner-main",
                    "lw-portal-banner-main--reverse",
                ]

        prev_button.on_click(lambda _event: _set_slide(active_slide["index"] - 1))
        next_button.on_click(lambda _event: _set_slide(active_slide["index"] + 1))

        banner_nav = pn.Row(
            prev_button,
            next_button,
            css_classes=["lw-portal-banner-nav-right"],
            margin=0,
        )
        banner = pn.Column(
            banner_main,
            banner_nav,
            css_classes=["lw-portal-banner"],
            sizing_mode="stretch_width",
            margin=(4, 0, 10, 0),
        )
        root.append(banner)
        _set_slide(0)

        doc = pn.state.curdoc
        if doc is not None and len(slides) > 1:

            def _advance_slide() -> None:
                _set_slide(active_slide["index"] + 1)

            doc.add_periodic_callback(_advance_slide, 5000)

    notebook_urls = notebook_urls_by_item()
    notebook_names = notebook_names_by_item()
    workspace = get_active_workspace()
    notebook_enabled = workspace is not None
    notebook_disabled_reason = (
        "Configure an active workspace first." if not notebook_enabled else None
    )

    mode_by_id = {mode.mode_id: mode for mode in QUICK_START_MODES}
    active_mode_id = (
        QUICK_START_DEFAULT_MODE
        if QUICK_START_DEFAULT_MODE in mode_by_id
        else QUICK_START_MODES[0].mode_id
    )
    quick_start_tab_sheet = InlineStyleSheet(
        css="""
        .bk-btn,
        .bk-btn:hover,
        .bk-btn:focus,
        .bk-btn:active {
          background: transparent !important;
          background-color: transparent !important;
          background-image: none !important;
          color: #1f2937 !important;
          box-shadow: none !important;
        }

        .bk-btn *,
        .bk-btn::before,
        .bk-btn::after,
        .bk-btn *::before,
        .bk-btn *::after {
          background: transparent !important;
          background-color: transparent !important;
          box-shadow: none !important;
          border: 0 !important;
          color: inherit !important;
        }
        """
    )

    def _quick_start_grid(mode_id: str) -> pn.viewable.Viewable:
        mode = mode_by_id[mode_id]
        cards = [
            render_card(
                ITEMS[item_id],
                show_lane_accent=False,
                notebook_urls=notebook_urls,
                notebook_names=notebook_names,
                notebook_enabled=notebook_enabled,
                notebook_disabled_reason=notebook_disabled_reason,
            )
            for item_id in mode.item_ids
            if item_id in ITEMS and ITEMS[item_id].status != "hidden"
        ]
        return pn.GridBox(
            *cards,
            ncols=3,
            css_classes=["lw-portal-grid"],
            sizing_mode="stretch_width",
        )

    quick_start_cards = pn.Column(
        _quick_start_grid(active_mode_id),
        css_classes=["lw-portal-quick-start-cards"],
        sizing_mode="stretch_width",
        margin=0,
    )

    def _apply_mode_animation() -> None:
        classes = [
            cls for cls in quick_start_cards.css_classes if cls != "lw-portal-qs-flip"
        ]
        classes.append("lw-portal-qs-flip")
        quick_start_cards.css_classes = classes

        document = pn.state.curdoc
        if document is None:
            quick_start_cards.css_classes = [
                cls for cls in classes if cls != "lw-portal-qs-flip"
            ]
            return

        def _clear_animation() -> None:
            quick_start_cards.css_classes = [
                cls
                for cls in quick_start_cards.css_classes
                if cls != "lw-portal-qs-flip"
            ]

        document.add_timeout_callback(_clear_animation, 430)

    mode_buttons: dict[str, pn.widgets.Button] = {}
    quick_start_mode_classes = {
        "user": "lw-portal-quick-start--user",
        "modeler": "lw-portal-quick-start--modeler",
        "experimentalist": "lw-portal-quick-start--experimentalist",
    }

    def _set_active_mode(mode_id: str) -> None:
        nonlocal active_mode_id
        if mode_id == active_mode_id:
            return
        active_mode_id = mode_id
        classes = [
            cls
            for cls in quick_start.css_classes
            if not cls.startswith("lw-portal-quick-start--")
        ]
        mode_cls = quick_start_mode_classes.get(mode_id)
        if mode_cls:
            classes.append(mode_cls)
        quick_start.css_classes = classes
        quick_start_cards[:] = [_quick_start_grid(mode_id)]
        _apply_mode_animation()
        for key, button in mode_buttons.items():
            classes = [
                cls
                for cls in button.css_classes
                if cls != "lw-portal-qs-top-tab--active"
            ]
            if key == mode_id:
                classes.append("lw-portal-qs-top-tab--active")
            button.css_classes = classes

    mode_tabs: list[pn.widgets.Button] = []
    mode_label_by_id = {
        "user": "User",
        "modeler": "Modeler",
        "experimentalist": "Experimentalist",
    }
    mode_class_by_id = {
        "user": "lw-portal-qs-top-tab--user",
        "modeler": "lw-portal-qs-top-tab--modeler",
        "experimentalist": "lw-portal-qs-top-tab--experimentalist",
    }
    for mode in QUICK_START_MODES:
        classes = ["lw-portal-qs-top-tab"]
        mode_class = mode_class_by_id.get(mode.mode_id)
        if mode_class:
            classes.append(mode_class)
        if mode.mode_id == active_mode_id:
            classes.append("lw-portal-qs-top-tab--active")
        button = pn.widgets.Button(
            name=mode_label_by_id.get(mode.mode_id, mode.title),
            button_type="default",
            margin=0,
            css_classes=classes,
            sizing_mode="fixed",
            width=124,
            height=28,
            stylesheets=[quick_start_tab_sheet],
        )
        mode_buttons[mode.mode_id] = button
        mode_tabs.append(button)
        button.on_click(lambda _event, mid=mode.mode_id: _set_active_mode(mid))

    mode_tabs_row = pn.Row(
        *mode_tabs,
        css_classes=["lw-portal-quick-start-tabs"],
        margin=(0, 0, 10, 0),
    )

    quick_start_main = pn.Column(
        pn.pane.HTML(
            '<div class="lw-portal-section-title">Quick Start</div>', margin=0
        ),
        quick_start_cards,
        css_classes=["lw-portal-quick-start-main"],
        margin=0,
        sizing_mode="stretch_width",
    )
    quick_start = pn.Column(
        pn.Column(
            mode_tabs_row,
            quick_start_main,
            css_classes=["lw-portal-quick-start-shell"],
            margin=0,
            sizing_mode="stretch_width",
        ),
        css_classes=["lw-portal-quick-start"],
        sizing_mode="stretch_width",
        margin=0,
    )
    quick_start.css_classes = [
        *quick_start.css_classes,
        quick_start_mode_classes.get(active_mode_id, "lw-portal-quick-start--modeler"),
    ]

    if workspace is None:
        root.append(
            pn.pane.HTML(
                (
                    '<div class="lw-portal-workspace-callout">'
                    "<div>"
                    '<div class="lw-portal-workspace-callout-title">Workspace setup required</div>'
                    '<div class="lw-portal-workspace-callout-copy">'
                    "Select or initialize a Larvaworld workspace before using notebook-based workflows "
                    "and other persistent portal features. Use the workspace control in the header to continue."
                    "</div>"
                    "</div>"
                    "</div>"
                ),
                margin=(0, 0, 14, 0),
                sizing_mode="stretch_width",
            )
        )
    root.append(quick_start)

    # Lanes
    for lane in LANES:
        lane_items = [
            ITEMS[item_id]
            for item_id in lane.item_ids
            if ITEMS[item_id].status != "hidden"
        ]
        root.append(
            render_lane(
                lane,
                items=lane_items,
                notebook_urls=notebook_urls,
                notebook_names=notebook_names,
                notebook_enabled=notebook_enabled,
                notebook_disabled_reason=notebook_disabled_reason,
            )
        )

    template.main.append(root)
    template.main.append(build_footer())
    return template
