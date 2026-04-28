from __future__ import annotations

import base64
import importlib.metadata as im
from html import escape
from pathlib import Path

import panel as pn

from larvaworld.portal.landing_registry import DOCS_ROOT, GITHUB_ISSUES, GITHUB_ROOT
from larvaworld.portal.registry_logic import (
    compute_badges,
    compute_primary_action,
    resolve_target,
)
from larvaworld.portal.registry_types import LandingItem, LaneSpec
from larvaworld.portal.workspace import get_active_workspace
from larvaworld.portal.workspace_ui import WorkspaceUiController


PORTAL_RAW_CSS = """
/* Scoped Portal styles (must not affect legacy dashboards). */
.lw-portal-root {
  max-width: 1240px;
  margin: 0 auto;
  padding: 16px 16px 56px 16px;
}

.lw-portal-root.lw-portal-dark {
  background: rgb(15, 23, 42);
  color: rgb(226, 232, 240);
  border-radius: 12px;
}

/* The template title bar is replaced by a custom top bar in #header-items. */
.app-header {
  display: none !important;
}

#header-items {
  margin-left: 0;
}

.lw-portal-topbar {
  display: flex;
  align-items: center !important;
  justify-content: space-between;
  gap: 16px;
  width: 100%;
  min-height: 60px;
}

.lw-portal-topbar > * {
  align-self: center !important;
}

.lw-portal-app-topbar {
  display: flex;
  align-items: center !important;
  gap: 16px;
  width: 100%;
  min-height: 60px;
}

.lw-portal-app-topbar > * {
  align-self: center !important;
}

.lw-portal-app-back {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 42px;
  height: 42px;
  border-radius: 999px;
  border: 2px solid rgba(17,17,17,0.78);
  background: rgba(255,255,255,0.98);
  color: #111111;
  text-decoration: none;
  font-weight: 700;
  line-height: 1;
  flex: 0 0 auto;
  box-shadow: none;
  box-sizing: border-box;
}

.lw-portal-app-back-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  line-height: 1;
  transform: translateX(-1px);
}

.lw-portal-app-back:hover {
  background: #f3f4f6;
  color: #111111;
  text-decoration: none;
}

.lw-portal-app-title {
  display: flex;
  align-items: center;
  font-size: 26px;
  font-weight: 650;
  color: rgba(17, 17, 17, 0.96);
  line-height: 1;
  min-height: 42px;
  margin: 0;
}

.lw-portal-header-left {
  min-width: 0;
}

.lw-portal-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
  color: inherit;
  min-width: 0;
}

.lw-portal-logo:hover {
  text-decoration: none;
}

.lw-portal-topbar .lw-portal-logo {
  color: rgba(255,255,255,0.96);
}

.lw-portal-logo-img {
  width: 58px;
  height: 58px;
  object-fit: contain;
  flex: 0 0 auto;
  background: #ffffff;
  border-radius: 10px;
  padding: 4px;
  box-sizing: border-box;
}

.lw-portal-logo-text {
  font-size: 22px;
  font-weight: 650;
  letter-spacing: 0.2px;
  margin: 0;
  line-height: 1;
}

.lw-portal-version-badge {
  display: inline-flex;
  align-items: center;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(0,0,0,0.18);
  background: rgba(0,0,0,0.04);
  color: rgba(0,0,0,0.78);
}

.lw-portal-topbar .lw-portal-version-badge {
  border-color: rgba(255,255,255,0.35);
  background: rgba(255,255,255,0.16);
  color: rgba(255,255,255,0.96);
}

.lw-portal-header-right-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-left: auto;
  flex: 0 0 auto;
  align-self: center !important;
}

.lw-portal-header-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 0 0 auto;
}

.lw-portal-workspace-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 34px;
  padding: 0 10px;
  border-radius: 10px;
  border: 1px solid rgba(0,0,0,0.18);
  background: #ffffff;
  color: #111111;
  white-space: nowrap;
}

.lw-portal-workspace-chip--missing {
  background: rgba(255,255,255,0.88);
}

.lw-portal-workspace-chip-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: rgba(0,0,0,0.56);
}

.lw-portal-workspace-chip-value {
  font-size: 12px;
  font-weight: 600;
}

.lw-portal-workspace-chip-path {
  font-size: 11px;
  color: rgba(0,0,0,0.56);
}

.lw-portal-topbar .lw-portal-workspace-chip {
  border-color: rgba(255,255,255,0.35);
  background: rgba(255,255,255,0.16);
  color: rgba(255,255,255,0.96);
}

.lw-portal-topbar .lw-portal-workspace-chip-label,
.lw-portal-topbar .lw-portal-workspace-chip-path {
  color: rgba(255,255,255,0.78);
}

.lw-portal-icon-link {
  width: 38px;
  min-width: 38px;
  height: 38px;
  padding: 0;
  flex: 0 0 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid rgba(17,17,17,0.14);
  background: #ffffff;
  color: #111111;
  text-decoration: none;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
}

.lw-portal-workspace-trigger-shell {
  position: relative;
  width: 22px;
  min-width: 22px;
  max-width: 22px;
  height: 22px;
  min-height: 22px;
  max-height: 22px;
  flex: 0 0 22px;
  margin: 0 !important;
}

.lw-portal-workspace-led {
  position: absolute;
  inset: 0;
  width: 22px;
  height: 22px;
  border-radius: 999px;
  border: 2px solid rgba(255,255,255,0.92);
  box-shadow: 0 0 0 1px rgba(17,17,17,0.12), 0 1px 2px rgba(15, 23, 42, 0.12);
  box-sizing: border-box;
}

.lw-portal-workspace-led--active {
  background: #2f8f4e;
  box-shadow: 0 0 0 1px rgba(17,17,17,0.12), 0 0 8px rgba(47, 143, 78, 0.28);
}

.lw-portal-workspace-led--inactive {
  background: #d94841;
}

.lw-portal-workspace-trigger-button,
.lw-portal-workspace-trigger-button .bk-btn,
.lw-portal-workspace-trigger-button button,
.lw-portal-workspace-trigger-button .mdc-button,
.lw-portal-workspace-trigger-button [class*="mdc-button"] {
  position: absolute !important;
  inset: 0 !important;
  width: 22px !important;
  min-width: 22px !important;
  max-width: 22px !important;
  height: 22px !important;
  min-height: 22px !important;
  max-height: 22px !important;
  margin: 0 !important;
  padding: 0 !important;
  opacity: 0 !important;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}

.lw-portal-icon-link:hover,
.lw-portal-topbar .lw-portal-settings-btn .bk-btn:hover {
  text-decoration: none;
}

.lw-portal-header-icon {
  width: 24px;
  height: 24px;
  object-fit: contain;
}

.lw-portal-settings-btn .bk-btn {
  text-decoration: none;
  padding: 0 12px;
  font-size: 13px;
  font-weight: 500;
}

.lw-portal-settings-btn {
  margin: 0 !important;
}

.lw-portal-settings-dropdown-wrap {
  position: relative;
  display: flex;
  display: inline-flex;
  align-items: center;
  width: 22px !important;
  min-width: 22px !important;
  max-width: 22px !important;
  flex: 0 0 22px;
  align-self: center !important;
  margin-right: 0 !important;
  overflow: visible;
}

.lw-portal-settings-panel {
  position: absolute;
  top: 40px;
  right: 0;
  width: min(388px, calc(100vw - 24px));
  max-width: calc(100vw - 24px);
  min-width: 0;
  padding: 0;
  border: 0;
  background: #ffffff;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.16);
  box-sizing: border-box;
  z-index: 30;
  color: #111111 !important;
  opacity: 1;
  pointer-events: auto;
}

.lw-portal-settings-title {
  font-size: 13px;
  font-weight: 650;
  margin: 0 0 6px 0;
  color: #111111;
}

.lw-portal-settings-title--dark {
  color: rgba(241,245,249,0.96) !important;
}

.lw-portal-field-label {
  font-size: 12px;
  font-weight: 600;
  color: rgba(17,17,17,0.76);
}

.lw-portal-field-label--dark {
  color: rgba(241,245,249,0.92) !important;
}

.lw-portal-settings-body {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  border: 1px solid rgba(0,0,0,0.22);
  border-radius: 12px;
  padding: 10px 12px;
  background: #ffffff !important;
  box-shadow: none;
  box-sizing: border-box;
  color: #111111 !important;
  opacity: 1;
  backdrop-filter: none;
}

.lw-portal-settings-row {
  font-size: 12px;
  color: rgba(0,0,0,0.72);
}

.lw-portal-settings-advanced {
  display: flex;
  align-items: center;
  gap: 6px;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  padding: 0 !important;
}

.lw-portal-settings-check {
  color: rgba(17,17,17,0.72);
  font-size: 12px;
  line-height: 1;
}

.lw-portal-settings-panel .bk,
.lw-portal-settings-panel label,
.lw-portal-settings-panel .bk-input-group,
.lw-portal-settings-panel .bk-form-group {
  color: #111111 !important;
}

.lw-portal-settings-body .bk,
.lw-portal-settings-body label,
.lw-portal-settings-body .bk-input-group,
.lw-portal-settings-body .bk-form-group {
  color: #111111 !important;
}

.lw-portal-settings-panel * {
  color: #111111 !important;
}

.lw-portal-settings-body * {
  color: #111111 !important;
}

.lw-portal-workspace-status {
  padding: 8px 10px;
  border-radius: 10px;
  font-size: 12px;
  line-height: 1.35;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(0,0,0,0.03);
  color: rgba(17,17,17,0.84);
}

.lw-portal-workspace-status--success {
  border-color: rgba(62,124,67,0.24);
  background: rgba(62,124,67,0.10);
}

.lw-portal-workspace-status--warning {
  border-color: rgba(176,112,33,0.28);
  background: rgba(245,161,66,0.12);
}

.lw-portal-workspace-status--danger {
  border-color: rgba(160,40,40,0.24);
  background: rgba(160,40,40,0.10);
}

.lw-portal-workspace-status-detail {
  margin-top: 4px;
  font-size: 11px;
  opacity: 0.84;
  word-break: break-word;
}

.lw-portal-workspace-callout {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(176,112,33,0.24);
  background: linear-gradient(135deg, rgba(245,161,66,0.16), rgba(245,161,66,0.08));
  color: #111111;
}

.lw-portal-workspace-callout-title {
  font-size: 15px;
  font-weight: 650;
  margin: 0 0 4px 0;
}

.lw-portal-workspace-callout-copy {
  font-size: 12px;
  line-height: 1.45;
  color: rgba(17,17,17,0.82);
}

.lw-portal-workspace-controls--dark .lw-portal-settings-title,
.lw-portal-workspace-controls--dark .lw-portal-settings-title--dark,
.lw-portal-workspace-controls--dark .lw-portal-field-label,
.lw-portal-workspace-controls--dark .lw-portal-field-label--dark,
.lw-portal-workspace-controls--dark label,
.lw-portal-workspace-controls--dark .bk-input-group label,
.lw-portal-workspace-controls--dark .bk-form-group label {
  color: rgba(241, 245, 249, 0.96) !important;
}

.lw-portal-workspace-status--theme-dark,
.lw-portal-workspace-status--theme-dark .lw-portal-workspace-status-detail,
.lw-portal-workspace-controls--dark .lw-portal-workspace-status,
.lw-portal-workspace-controls--dark .lw-portal-workspace-status *,
.lw-portal-workspace-controls--dark .lw-portal-workspace-status-detail {
  color: rgba(241, 245, 249, 0.94) !important;
}

.lw-portal-workspace-status--neutral.lw-portal-workspace-status--theme-dark,
.lw-portal-workspace-controls--dark .lw-portal-workspace-status {
  border-color: rgba(245, 161, 66, 0.28);
  background: rgba(255,255,255,0.08);
}

.lw-portal-workspace-status--success.lw-portal-workspace-status--theme-dark,
.lw-portal-workspace-controls--dark .lw-portal-workspace-status--success {
  border-color: rgba(134, 239, 172, 0.32);
  background: rgba(22, 101, 52, 0.28);
}

.lw-portal-workspace-status--warning.lw-portal-workspace-status--theme-dark,
.lw-portal-workspace-controls--dark .lw-portal-workspace-status--warning {
  border-color: rgba(245, 161, 66, 0.34);
  background: rgba(245, 161, 66, 0.16);
}

.lw-portal-workspace-status--danger.lw-portal-workspace-status--theme-dark,
.lw-portal-workspace-controls--dark .lw-portal-workspace-status--danger {
  border-color: rgba(248, 113, 113, 0.34);
  background: rgba(127, 29, 29, 0.24);
}

.lw-portal-workspace-controls {
  width: 100%;
  max-width: 100%;
}

.lw-portal-workspace-actions {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  column-gap: 12px;
  row-gap: 10px;
  align-items: stretch;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}

.lw-portal-workspace-path-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  align-items: stretch;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}

.lw-portal-workspace-path-row > :first-child {
  width: 100% !important;
  min-width: 0;
  max-width: 100%;
  box-sizing: border-box;
  margin: 0 !important;
}

.lw-portal-workspace-actions > * {
  width: 100% !important;
  min-width: 0;
  max-width: 100%;
}

.lw-portal-workspace-input {
  width: 100% !important;
  max-width: 100%;
  box-sizing: border-box;
}

.lw-portal-workspace-actions .bk-btn,
.lw-portal-workspace-actions button,
.lw-portal-workspace-browse-btn .bk-btn,
.lw-portal-workspace-browse-btn button {
  margin: 0 !important;
}

.lw-portal-workspace-actions .bk-btn,
.lw-portal-workspace-actions button {
  width: 100% !important;
}

.lw-portal-workspace-input .bk,
.lw-portal-workspace-input .bk-input-group,
.lw-portal-workspace-input .bk-input-group input {
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}

.lw-portal-workspace-browse-btn .bk-btn,
.lw-portal-workspace-browse-btn button,
.lw-portal-workspace-browse-btn .bk-btn:hover,
.lw-portal-workspace-browse-btn button:hover,
.lw-portal-workspace-browse-btn .bk-btn:focus,
.lw-portal-workspace-browse-btn button:focus,
.lw-portal-workspace-browse-btn .bk-btn:active,
.lw-portal-workspace-browse-btn button:active {
  min-height: 36px !important;
  padding: 0 10px !important;
  color: #ffffff !important;
}

.lw-portal-workspace-browse-btn .bk-btn *,
.lw-portal-workspace-browse-btn button *,
.lw-portal-workspace-browse-btn .mdc-button__label,
.lw-portal-workspace-browse-btn [class*="mdc-button"] *,
.lw-portal-workspace-browse-btn span {
  color: #ffffff !important;
  fill: #ffffff !important;
}

.lw-portal-settings-panel .lw-portal-workspace-input .bk-input-group input {
  width: 100% !important;
  min-height: 36px;
  padding: 7px 10px !important;
  border: 1px solid rgba(0,0,0,0.18) !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  box-shadow: none !important;
}

.lw-portal-settings-panel .bk-input-group,
.lw-portal-settings-panel .bk-form-group,
.lw-portal-settings-panel .bk-input-group input {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}

.lw-portal-root.lw-portal-dark .lw-portal-section-title,
.lw-portal-root.lw-portal-dark .lw-portal-card-title {
  color: rgb(241, 245, 249);
}

.lw-portal-root.lw-portal-dark .lw-portal-card {
  border-color: rgba(148, 163, 184, 0.45);
  background: rgba(30, 41, 59, 0.88);
  box-shadow: none;
}

.lw-portal-root.lw-portal-dark .lw-portal-card-subtitle,
.lw-portal-root.lw-portal-dark .lw-portal-settings-row {
  color: rgba(203, 213, 225, 0.9);
}

.lw-portal-root.lw-portal-dark .lw-portal-badge,
.lw-portal-root.lw-portal-dark .lw-portal-version-badge {
  border-color: rgba(148, 163, 184, 0.42);
  background: rgba(15, 23, 42, 0.72);
  color: rgb(226, 232, 240);
}

.lw-portal-section-title {
  margin: 18px 0 10px 0;
  font-size: 18px;
  font-weight: 650;
}

.lw-portal-banner {
  position: relative;
  height: 300px;
  border: 1px solid rgba(148, 163, 184, 0.35);
  border-radius: 14px;
  background: rgba(17, 24, 39, 0.96);
  box-shadow: 0 2px 10px rgba(0,0,0,0.18);
  overflow: hidden;
}

.lw-portal-banner-main {
  display: flex;
  align-items: stretch;
  gap: 0;
  height: 300px;
}

.lw-portal-banner-main--reverse {
  flex-direction: row-reverse;
}

.lw-portal-banner-media {
  flex: 0 0 74%;
  min-width: 0;
  min-height: 300px;
}

.lw-portal-banner-gif {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
  min-height: 300px;
}

.lw-portal-banner-copy {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 8px;
  padding: 16px 58px 16px 16px;
  height: 300px;
  flex: 1 1 26%;
  overflow: hidden;
}

.lw-portal-banner-title {
  font-size: 18px;
  font-weight: 650;
  color: rgba(241, 245, 249, 0.98);
}

.lw-portal-banner-description {
  font-size: 14px;
  line-height: 1.45;
  color: rgba(203, 213, 225, 0.92);
  max-width: 72ch;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.lw-portal-banner-link {
  display: inline-flex;
  width: fit-content;
  align-items: center;
  justify-content: center;
  padding: 6px 11px;
  border-radius: 9px;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
  border: 1px solid rgba(245, 161, 66, 0.72);
  background: rgba(245, 161, 66, 0.18);
  color: rgba(255, 237, 213, 0.98);
}

.lw-portal-banner-link:hover {
  background: rgba(245, 161, 66, 0.28);
}

.lw-portal-banner-nav-right {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  position: absolute;
  right: 14px;
  bottom: 14px;
  gap: 8px;
  z-index: 3;
  margin: 0 !important;
}

.lw-portal-banner-nav-btn .bk-btn {
  width: 30px !important;
  min-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.62);
  background: rgba(15, 23, 42, 0.86);
  color: rgba(241, 245, 249, 0.96);
  font-size: 16px;
  line-height: 1;
  padding: 0 !important;
}

.lw-portal-banner-nav-btn .bk-btn:hover {
  background: rgba(30, 41, 59, 0.94);
}

.lw-portal-banner-nav-btn--left {
  position: static;
}

.lw-portal-banner-nav-btn--next {
  position: static;
}

.lw-portal-root.lw-portal-dark .lw-portal-banner-nav-btn .bk-btn {
  border-color: rgba(148, 163, 184, 0.62);
  background: rgba(15, 23, 42, 0.86);
  color: rgba(241, 245, 249, 0.96);
}

.lw-portal-root.lw-portal-dark .lw-portal-banner-nav-btn .bk-btn:hover {
  background: rgba(30, 41, 59, 0.92);
}

.lw-portal-quick-start {
  margin: 30px 0 18px 0;
  padding: 8px 12px 14px 10px;
  border-radius: 14px;
  /* Source-of-truth color for the Quick Start lane. */
  --lw-quick-start-bg: rgba(0,0,0,0.08);
  background: rgba(0,0,0,0.08);
  border-left: 4px solid #f5a142;
  overflow: visible;
}

.lw-portal-quick-start--user { border-left-color: #f5a142; }
.lw-portal-quick-start--modeler { border-left-color: #f5a142; }
.lw-portal-quick-start--experimentalist { border-left-color: #f5a142; }

.lw-portal-root.lw-portal-dark .lw-portal-quick-start {
  background: rgba(2, 6, 23, 0.5);
}

.lw-portal-quick-start-shell {
  display: block;
  position: relative;
  overflow: visible;
}

.lw-portal-quick-start-main {
  flex: 1 1 auto;
  min-width: 0;
}

.lw-portal-quick-start-tabs {
  position: absolute;
  top: -35px;
  right: 8px;
  z-index: 7;
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  gap: 10px;
  margin: 0 !important;
}

.lw-portal-qs-top-tab,
.lw-portal-qs-top-tab.bk-btn,
button.lw-portal-qs-top-tab,
.lw-portal-qs-top-tab button,
.lw-portal-qs-top-tab .bk-btn {
  width: 124px !important;
  min-width: 124px !important;
  max-width: 124px !important;
  height: 28px !important;
  min-height: 28px !important;
  max-height: 28px !important;
  border-radius: 10px 10px 0 0;
  border: 1px solid rgba(148, 163, 184, 0.75) !important;
  border-bottom: 0 !important;
  color: #1f2937 !important;
  font-weight: 600;
  text-align: center;
  font-size: 11px;
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
  appearance: none !important;
  -webkit-appearance: none !important;
  padding: 0 8px !important;
  white-space: nowrap !important;
  letter-spacing: 0.1px !important;
  background: var(--lw-quick-start-bg, rgba(0,0,0,0.08)) !important;
  background-color: var(--lw-quick-start-bg, rgba(0,0,0,0.08)) !important;
  background-image: none !important;
  box-shadow: none !important;
  transition: transform 0.14s ease, filter 0.14s ease, box-shadow 0.14s ease;
  margin-bottom: 0 !important;
}

/* Force the real clickable element (Bokeh/Material internals) to use lane gray. */
.lw-portal-qs-top-tab .bk-btn,
.lw-portal-qs-top-tab button,
.lw-portal-qs-top-tab .mdc-button,
.lw-portal-qs-top-tab [class*="mdc-button"] {
  background: transparent !important;
  background-color: transparent !important;
  background-image: none !important;
  color: #1f2937 !important;
}

.lw-portal-qs-top-tab > span,
.lw-portal-qs-top-tab.bk-btn > span,
button.lw-portal-qs-top-tab > span,
.lw-portal-qs-top-tab button > span,
.lw-portal-qs-top-tab .bk-btn > span {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
  color: inherit !important;
  transform: none !important;
}

/* Remove white inner capsule from Material/Bokeh button internals. */
.lw-portal-qs-top-tab .bk-btn *,
.lw-portal-qs-top-tab.bk-btn *,
button.lw-portal-qs-top-tab * {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
  color: inherit !important;
}

.lw-portal-qs-top-tab .mdc-button__label,
.lw-portal-qs-top-tab .mdc-button__ripple,
.lw-portal-qs-top-tab .mdc-button__touch {
  background: transparent !important;
  background-color: transparent !important;
}

.lw-portal-qs-top-tab .bk-btn::before,
.lw-portal-qs-top-tab .bk-btn::after,
.lw-portal-qs-top-tab.bk-btn::before,
.lw-portal-qs-top-tab.bk-btn::after,
button.lw-portal-qs-top-tab::before,
button.lw-portal-qs-top-tab::after {
  background: transparent !important;
  box-shadow: none !important;
}

.lw-portal-qs-top-tab--user,
.lw-portal-qs-top-tab--user.bk-btn,
button.lw-portal-qs-top-tab--user,
.lw-portal-qs-top-tab--user button,
.lw-portal-qs-top-tab--user .bk-btn {
  color: #1f2937 !important;
  border-color: rgba(148, 163, 184, 0.75) !important;
}
.lw-portal-qs-top-tab--modeler,
.lw-portal-qs-top-tab--modeler.bk-btn,
button.lw-portal-qs-top-tab--modeler,
.lw-portal-qs-top-tab--modeler button,
.lw-portal-qs-top-tab--modeler .bk-btn {
  color: #1f2937 !important;
  border-color: rgba(148, 163, 184, 0.75) !important;
}
.lw-portal-qs-top-tab--experimentalist,
.lw-portal-qs-top-tab--experimentalist.bk-btn,
button.lw-portal-qs-top-tab--experimentalist,
.lw-portal-qs-top-tab--experimentalist button,
.lw-portal-qs-top-tab--experimentalist .bk-btn {
  color: #1f2937 !important;
  border-color: rgba(148, 163, 184, 0.75) !important;
}

.lw-portal-qs-top-tab:hover,
.lw-portal-qs-top-tab.bk-btn:hover,
button.lw-portal-qs-top-tab:hover,
.lw-portal-qs-top-tab button:hover,
.lw-portal-qs-top-tab .bk-btn:hover {
  filter: brightness(1.03);
}

.lw-portal-qs-top-tab--active,
.lw-portal-qs-top-tab--active.bk-btn,
button.lw-portal-qs-top-tab--active,
.lw-portal-qs-top-tab--active button,
.lw-portal-qs-top-tab--active .bk-btn {
  transform: none !important;
  box-shadow: 0 -4px 8px -5px rgba(15, 23, 42, 0.75) !important;
  filter: brightness(1.04);
  border-color: #f5a142 !important;
}

.lw-portal-quick-start-cards {
  flex: 1 1 auto;
}

.lw-portal-quick-start-cards.lw-portal-qs-flip .lw-portal-card {
  animation: lwPortalFlipRotate 420ms ease;
  transform-origin: center center;
}

@keyframes lwPortalFlipRotate {
  0% { transform: rotateY(0deg) scale(1.0); opacity: 1; }
  50% { transform: rotateY(90deg) scale(0.98); opacity: 0.55; }
  100% { transform: rotateY(0deg) scale(1.0); opacity: 1; }
}

.lw-portal-grid {
  /* Let Panel control layout; we only suggest spacing. */
  gap: 14px;
}

.lw-portal-card {
  position: relative;
  display: flex;
  flex-direction: column;
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 14px;
  padding: 14px 14px 4px 14px;
  background: rgba(255,255,255,0.96);
  box-shadow: 0 1px 8px rgba(0,0,0,0.05);
  height: 186px;
  overflow: hidden;
  cursor: pointer;
  transition: transform 140ms ease, box-shadow 140ms ease;
}

.lw-portal-card:hover {
  transform: scale(1.015);
  box-shadow: 0 10px 18px rgba(0,0,0,0.10);
}

.lw-portal-card-link-overlay {
  display: block;
  width: 100%;
  height: 100%;
  border-radius: 14px;
  text-decoration: none;
  cursor: pointer;
}

.lw-portal-card--planned {
  opacity: 0.86;
}

.lw-portal-card--lane-simulate {
  border-top: 3px solid #b5c2b0;
}

.lw-portal-card--lane-data {
  border-top: 3px solid #b0b4c2;
}

.lw-portal-card--lane-models {
  border-top: 3px solid #c1b0c2;
}

.lw-portal-card--lane-eval {
  border-top: 3px solid #b1b2de;
}

.lw-portal-card--lane-simulate:hover {
  box-shadow: 0 10px 18px rgba(0,0,0,0.08), 0 0 0 9999px rgba(181,194,176,0.14) inset;
}

.lw-portal-card--lane-data:hover {
  box-shadow: 0 10px 18px rgba(0,0,0,0.08), 0 0 0 9999px rgba(176,180,194,0.14) inset;
}

.lw-portal-card--lane-models:hover {
  box-shadow: 0 10px 18px rgba(0,0,0,0.08), 0 0 0 9999px rgba(193,176,194,0.14) inset;
}

.lw-portal-card--lane-eval:hover {
  box-shadow: 0 10px 18px rgba(0,0,0,0.08), 0 0 0 9999px rgba(177,178,222,0.14) inset;
}

.lw-portal-card-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 8px;
}

.lw-portal-badge {
  display: inline-flex;
  align-items: center;
  padding: 1px 7px;
  border-radius: 999px;
  font-size: 11px;
  line-height: 1.2;
  border: 1px solid rgba(0,0,0,0.18);
  background: rgba(0,0,0,0.04);
}

.lw-portal-badge--under-construction {
  border-color: rgba(239,108,0,0.55);
  background: rgba(239,108,0,0.10);
}

.lw-portal-card-title {
  font-size: 16px;
  font-weight: 650;
  margin: 0 0 6px 0;
}

.lw-portal-card-subtitle {
  font-size: 13px;
  line-height: 1.35;
  margin: 0 0 8px 0;
  min-height: calc(1.35em * 3);
  max-height: calc(1.35em * 3);
  overflow: hidden;
  color: rgba(0,0,0,0.72);
}

.lw-portal-actions {
  position: relative;
  z-index: 4;
  display: flex;
  align-items: center;
  margin-top: auto;
  gap: 10px;
}

.lw-portal-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 8px 12px;
  border-radius: 10px;
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
  border: 1px solid rgba(25,118,210,0.35);
  background: rgba(25,118,210,0.10);
  color: rgb(25,118,210);
}

.lw-portal-btn:hover {
  background: rgba(25,118,210,0.14);
}

.lw-portal-btn--learn-more {
  padding: 4px 8px;
  font-size: 11px;
}

.lw-portal-btn--notebook-simulate {
  border-color: rgba(181,194,176,0.92);
  background: rgba(181,194,176,0.24);
  color: rgb(46, 60, 42);
}

.lw-portal-btn--notebook-data {
  border-color: rgba(176,180,194,0.92);
  background: rgba(176,180,194,0.24);
  color: rgb(44, 48, 61);
}

.lw-portal-btn--notebook-models {
  border-color: rgba(193,176,194,0.92);
  background: rgba(193,176,194,0.24);
  color: rgb(64, 47, 66);
}

.lw-portal-btn--notebook-eval {
  border-color: rgba(177,178,222,0.92);
  background: rgba(177,178,222,0.24);
  color: rgb(43, 44, 84);
}

.lw-portal-btn--notebook-simulate:hover {
  background: rgba(181,194,176,0.34);
}

.lw-portal-btn--notebook-data:hover {
  background: rgba(176,180,194,0.34);
}

.lw-portal-btn--notebook-models:hover {
  background: rgba(193,176,194,0.34);
}

.lw-portal-btn--notebook-eval:hover {
  background: rgba(177,178,222,0.34);
}

.lw-portal-btn--disabled {
  border-color: rgba(0,0,0,0.14);
  background: rgba(0,0,0,0.04);
  color: rgba(0,0,0,0.38);
  pointer-events: none;
}

.lw-portal-footer-shell {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 40;
}

.lw-portal-footer-bar {
  width: 100%;
  min-height: 24px;
  padding: 4px 10px;
  background: #f5a142;
  color: #111111;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  font-size: 11px;
  box-sizing: border-box;
}

.lw-portal-footer-link {
  color: #111111;
  text-decoration: underline;
}

.lw-portal-footer-link:hover {
  color: #111111;
}
""".strip()


def _load_icon_data_uri(filename: str, mime_type: str) -> str:
    icon_path = Path(__file__).with_name("icons") / filename
    try:
        encoded = base64.b64encode(icon_path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    return f"data:{mime_type};base64,{encoded}"


def _resolve_portal_version() -> str:
    try:
        return im.version("larvaworld")
    except Exception:
        return "dev"


_LOGO_DATA_URI = _load_icon_data_uri("LarvaWorld_logo.png", "image/png")
_RTD_ICON_DATA_URI = _load_icon_data_uri("RTD_logo.svg", "image/svg+xml")
_GITHUB_ICON_DATA_URI = _load_icon_data_uri("github_logo.svg", "image/svg+xml")
_PORTAL_VERSION = _resolve_portal_version()


def _portal_logo_html(*, version: str) -> str:
    logo_img = ""
    if _LOGO_DATA_URI:
        logo_img = f'<img class="lw-portal-logo-img" src="{_LOGO_DATA_URI}" alt="Larvaworld logo"/>'

    return (
        '<a class="lw-portal-logo" href="/landing" '
        "onclick=\"if (window.location.pathname !== '/landing' && "
        "window.location.pathname !== '/') { return confirm('Leave this page?\\n\\nReturning "
        "to the landing page will reset the current view. Any unsaved selections or progress "
        "may be lost.'); } return true;\">"
        f"{logo_img}"
        '<span class="lw-portal-logo-text">Larvaworld</span>'
        f'<span class="lw-portal-version-badge">v{escape(version)}</span>'
        "</a>"
    )


def _header_links_html() -> str:
    docs_icon = ""
    if _RTD_ICON_DATA_URI:
        docs_icon = (
            f'<img class="lw-portal-header-icon" src="{_RTD_ICON_DATA_URI}" '
            'alt="Read the Docs logo"/>'
        )

    github_icon = ""
    if _GITHUB_ICON_DATA_URI:
        github_icon = (
            f'<img class="lw-portal-header-icon" src="{_GITHUB_ICON_DATA_URI}" '
            'alt="GitHub logo"/>'
        )

    return (
        '<div class="lw-portal-header-right">'
        f'<a class="lw-portal-icon-link" href="{escape(DOCS_ROOT)}" '
        'target="_blank" rel="noopener noreferrer" title="Read the Docs">'
        f"{docs_icon}"
        "</a>"
        f'<a class="lw-portal-icon-link" href="{escape(GITHUB_ROOT)}" '
        'target="_blank" rel="noopener noreferrer" title="GitHub">'
        f"{github_icon}"
        "</a>"
        "</div>"
    )


def _badge_html(badge: str) -> str:
    cls = "lw-portal-badge"
    if badge.strip().lower() in {"under construction", "planned"}:
        cls += " lw-portal-badge--under-construction"
    return f'<span class="{cls}">{escape(badge)}</span>'


def _button_html(
    *,
    label: str,
    href: str | None,
    enabled: bool,
    extra_classes: tuple[str, ...] = (),
    tooltip: str | None = None,
) -> str:
    normalized_label = label.strip().lower()
    button_classes = ["lw-portal-btn"]
    if normalized_label in {"learn more", "notebook"}:
        button_classes.append("lw-portal-btn--learn-more")
    if normalized_label == "notebook":
        button_classes.append("lw-portal-btn--notebook")
    button_classes.extend(extra_classes)
    class_attr = " ".join(button_classes)
    title_attr = f' title="{escape(tooltip)}"' if tooltip else ""

    if not enabled or not href:
        return f'<span class="{class_attr} lw-portal-btn--disabled"{title_attr}>{escape(label)}</span>'

    attrs = ""
    if normalized_label == "notebook":
        attrs = ' target="_blank" rel="noopener noreferrer"'
    elif href.startswith("http://") or href.startswith("https://"):
        attrs = ' target="_blank" rel="noopener noreferrer"'

    return f'<a class="{class_attr}" href="{escape(href)}"{attrs}{title_attr}>{escape(label)}</a>'


def _subtitle_html(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [""]
    return "<br/>".join(escape(line) for line in lines[:3])


def build_footer() -> pn.viewable.Viewable:
    html = (
        '<div class="lw-portal-footer-shell"><div class="lw-portal-footer-bar">'
        "<span>&copy; Larvaworld</span>"
        f"<span>v{escape(_PORTAL_VERSION)}</span>"
        "<span>University of Cologne</span>"
        f'<a class="lw-portal-footer-link" href="{escape(DOCS_ROOT)}" '
        'target="_blank" rel="noopener noreferrer">Docs</a>'
        f'<a class="lw-portal-footer-link" href="{escape(GITHUB_ROOT)}" '
        'target="_blank" rel="noopener noreferrer">GitHub</a>'
        f'<a class="lw-portal-footer-link" href="{escape(GITHUB_ISSUES)}" '
        'target="_blank" rel="noopener noreferrer">Issues</a>'
        '<a class="lw-portal-footer-link" href="mailto:p.sakagiannis@uni-koeln.de">'
        "Contact</a>"
        "</div></div>"
    )
    return pn.pane.HTML(html, margin=0, sizing_mode="stretch_width")


def build_template_header() -> pn.viewable.Viewable:
    workspace_ui = WorkspaceUiController()
    left = pn.pane.HTML(
        _portal_logo_html(version=_PORTAL_VERSION),
        margin=0,
        css_classes=["lw-portal-header-left"],
    )
    links = pn.pane.HTML(
        _header_links_html(),
        margin=0,
    )
    workspace_controls = workspace_ui.build_controls()
    settings_body = pn.Column(
        workspace_controls,
        css_classes=["lw-portal-settings-body"],
        sizing_mode="stretch_width",
        margin=0,
    )
    settings_panel = pn.Column(
        settings_body,
        visible=get_active_workspace() is None,
        css_classes=["lw-portal-settings-panel"],
        margin=0,
    )

    def _toggle_settings(_: object) -> None:
        settings_panel.visible = not settings_panel.visible

    workspace_ui.trigger_button.on_click(_toggle_settings)

    settings_dropdown = pn.Column(
        workspace_ui.trigger_view,
        settings_panel,
        css_classes=["lw-portal-settings-dropdown-wrap"],
        margin=0,
        sizing_mode="fixed",
        width_policy="min",
    )
    right = pn.Row(
        links,
        settings_dropdown,
        css_classes=["lw-portal-header-right-wrap"],
        margin=0,
        sizing_mode="fixed",
        width_policy="min",
    )
    header_row = pn.Row(
        left,
        pn.Spacer(),
        pn.Spacer(sizing_mode="stretch_width"),
        right,
        css_classes=["lw-portal-topbar"],
        sizing_mode="stretch_width",
        margin=0,
    )
    return header_row


def build_app_header(
    *, title: str, back_href: str = "/landing"
) -> pn.viewable.Viewable:
    workspace_ui = WorkspaceUiController()
    back_button = pn.pane.HTML(
        (
            f'<a class="lw-portal-app-back" href="{escape(back_href)}" '
            'title="Back to landing" aria-label="Back to landing">'
            '<span class="lw-portal-app-back-icon">&#8249;</span></a>'
        ),
        margin=0,
    )
    title_pane = pn.pane.HTML(
        f'<div class="lw-portal-app-title">{escape(title)}</div>',
        margin=0,
    )
    return pn.Row(
        back_button,
        title_pane,
        pn.Spacer(sizing_mode="stretch_width"),
        workspace_ui.chip_pane,
        css_classes=["lw-portal-app-topbar"],
        sizing_mode="stretch_width",
        margin=0,
    )


def render_card(
    item: LandingItem,
    *,
    show_lane_accent: bool = True,
    notebook_urls: dict[str, str] | None = None,
    notebook_names: dict[str, str] | None = None,
    notebook_enabled: bool = True,
    notebook_disabled_reason: str | None = None,
) -> pn.viewable.Viewable:
    action = compute_primary_action(item)
    badges = compute_badges(item)
    card_href = resolve_target(item) or f"/{item.id}"

    card_classes = ["lw-portal-card"]
    lane_classes = {
        "simulate": "lw-portal-card--lane-simulate",
        "data": "lw-portal-card--lane-data",
        "models": "lw-portal-card--lane-models",
        "eval": "lw-portal-card--lane-eval",
    }
    lane_class = lane_classes.get(item.lane)
    if lane_class and show_lane_accent:
        card_classes.append(lane_class)
    if item.status == "planned" or item.kind == "placeholder":
        card_classes.append("lw-portal-card--planned")

    badges_html = "".join(_badge_html(b) for b in badges)

    primary_action_html = _button_html(
        label="Learn more",
        href=action.href,
        enabled=action.enabled,
    )
    notebook_href = notebook_urls.get(item.id) if notebook_urls else None
    notebook_action_html = ""
    notebook_available = bool(
        notebook_href or (notebook_names is not None and item.id in notebook_names)
    )
    if notebook_available:
        notebook_lane_classes = {
            "simulate": "lw-portal-btn--notebook-simulate",
            "data": "lw-portal-btn--notebook-data",
            "models": "lw-portal-btn--notebook-models",
            "eval": "lw-portal-btn--notebook-eval",
        }
        notebook_lane_class = notebook_lane_classes.get(item.lane)
        extra_classes: tuple[str, ...] = ()
        if notebook_lane_class:
            extra_classes = (notebook_lane_class,)
        notebook_action_html = _button_html(
            label="Notebook",
            href=notebook_href if notebook_enabled else None,
            enabled=notebook_enabled and bool(notebook_href),
            extra_classes=extra_classes,
            tooltip=(
                notebook_disabled_reason
                if not notebook_enabled and notebook_disabled_reason
                else (
                    f"Open notebook: {notebook_names[item.id]}"
                    if notebook_names and item.id in notebook_names
                    else "Open notebook"
                )
            ),
        )

    actions_html = (
        '<div class="lw-portal-actions">'
        + primary_action_html
        + notebook_action_html
        + "</div>"
    )
    show_actions = bool(primary_action_html or notebook_action_html)

    overlay_attrs = ""
    if card_href.startswith("http://") or card_href.startswith("https://"):
        overlay_attrs = ' target="_blank" rel="noopener noreferrer"'
    overlay_style = (
        "position:absolute;inset:0;display:block;width:100%;height:100%;"
        "border-radius:14px;text-decoration:none;cursor:pointer;"
    )
    overlay_html = (
        f'<a class="lw-portal-card-link-overlay" style="{overlay_style}" '
        f'href="{escape(card_href)}"{overlay_attrs} '
        f'aria-label="Open {escape(item.title)}"></a>'
    )
    actions_pane = pn.pane.HTML(
        actions_html,
        margin=0,
        visible=show_actions,
        styles={"position": "relative", "z-index": "4"},
    )
    overlay_pane = pn.pane.HTML(
        overlay_html,
        margin=0,
        visible=bool(card_href),
        styles={
            "position": "absolute",
            "inset": "0",
            "width": "100%",
            "height": "100%",
            "z-index": "3",
            "pointer-events": "auto",
        },
    )

    body = pn.Column(
        pn.pane.HTML(
            f'<div class="lw-portal-card-badges">{badges_html}</div>', margin=0
        ),
        pn.pane.HTML(
            f'<div class="lw-portal-card-title">{escape(item.title)}</div>', margin=0
        ),
        pn.pane.HTML(
            f'<div class="lw-portal-card-subtitle">{_subtitle_html(item.subtitle)}</div>',
            margin=0,
        ),
        actions_pane,
        overlay_pane,
        css_classes=card_classes,
        styles={"position": "relative"},
        margin=0,
        sizing_mode="stretch_width",
    )
    return body


def render_lane(
    lane: LaneSpec,
    *,
    items: list[LandingItem],
    notebook_urls: dict[str, str] | None = None,
    notebook_names: dict[str, str] | None = None,
    notebook_enabled: bool = True,
    notebook_disabled_reason: str | None = None,
) -> pn.viewable.Viewable:
    title = pn.pane.HTML(
        f'<div class="lw-portal-section-title">{escape(lane.title)}</div>', margin=0
    )
    cards = [
        render_card(
            item,
            notebook_urls=notebook_urls,
            notebook_names=notebook_names,
            notebook_enabled=notebook_enabled,
            notebook_disabled_reason=notebook_disabled_reason,
        )
        for item in items
    ]
    grid = pn.pane.HTML("", visible=False)  # placeholder to keep types simple
    if cards:
        grid = pn.GridBox(
            *cards, ncols=4, css_classes=["lw-portal-grid"], sizing_mode="stretch_width"
        )

    content = pn.Column(title, grid, sizing_mode="stretch_width", margin=0)
    if not lane.collapsed_by_default:
        return content

    # Collapsed lane (demo/tutorials) uses an accordion to avoid distracting the main workflows.
    return pn.Accordion((lane.title, content), active=[], sizing_mode="stretch_width")
