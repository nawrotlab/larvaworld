"""Unit tests for ScreenTextFont's line wrapping and spacing."""

from __future__ import annotations

import pytest


@pytest.fixture
def pygame_ready():
    import pygame

    pygame.init()
    yield
    pygame.quit()


@pytest.mark.fast
class TestScreenTextFontWrapping:
    def test_no_wrapping_when_max_text_width_unset(self, pygame_ready):
        from larvaworld.lib.screen.rendering import ScreenTextFont

        f = ScreenTextFont(text="a very long line with several words in it")
        assert f.text_lines == ["a very long line with several words in it"]

    def test_wraps_long_line_to_fit_max_width(self, pygame_ready):
        from larvaworld.lib.screen.rendering import ScreenTextFont

        f = ScreenTextFont(
            text="This is a fairly long line that must wrap across several rendered lines",
            font_size=20,
            max_text_width=200,
        )
        lines = f.text_lines
        assert len(lines) > 1
        for line in lines:
            assert f.font.size(line)[0] <= 200

    def test_splits_single_oversized_token_with_no_spaces(self, pygame_ready):
        """
        A refID-like token (underscore-joined, no spaces) that alone exceeds
        max_text_width can't be word-wrapped -- must fall back to
        character-level splitting instead of overflowing the frame.
        """
        from larvaworld.lib.screen.rendering import ScreenTextFont

        f = ScreenTextFont(
            text="replay_super_long_experiment_name_that_is_quite_verbose",
            font_size=20,
            max_text_width=150,
        )
        lines = f.text_lines
        assert len(lines) > 1
        for line in lines:
            assert f.font.size(line)[0] <= 150
        assert "".join(lines) == f.text

    def test_line_spacing_scales_with_font_size(self, pygame_ready):
        from larvaworld.lib.screen.rendering import ScreenTextFont

        f = ScreenTextFont(
            text="line one\nline two\nline three",
            font_size=40,
            text_centre=(100, 100),
        )
        f.render_text()
        centers_y = [r.centery for r in f.text_font_r]
        gaps = {b - a for a, b in zip(centers_y, centers_y[1:])}
        assert gaps == {int(40 * f.line_spacing_scale)}


@pytest.mark.fast
class TestScreenTextBoxRectAutoWidth:
    def test_derives_max_text_width_from_frame_rect(self, pygame_ready):
        import pygame as pg
        from larvaworld.lib.screen.rendering import ScreenTextBoxRect

        rect = pg.Rect(0, 0, 300, 200)
        box = ScreenTextBoxRect(text="hello", frame_rect=rect, font_size=20)
        assert box.max_text_width == int(300 * 0.92)

    def test_explicit_max_text_width_is_not_overridden(self, pygame_ready):
        import pygame as pg
        from larvaworld.lib.screen.rendering import ScreenTextBoxRect

        rect = pg.Rect(0, 0, 300, 200)
        box = ScreenTextBoxRect(
            text="hello", frame_rect=rect, font_size=20, max_text_width=50
        )
        assert box.max_text_width == 50
