"""
Unit tests for model value grid modules.

Tests cover:
- Grid: Basic grid dimensions and properties
- ValueGrid: Value storage, retrieval, cell operations
- FoodGrid: Food-specific color mapping
- GaussianValueLayer: Analytical odor gradients
- ThermoScape: Temperature gradient computation

Strategy:
- Mock minimal model/space context
- Test grid operations without full simulation
- Focus on value computation logic
- Avoid rendering/visualization code
"""

import numpy as np
import pytest
from unittest.mock import Mock

from larvaworld.lib.model.envs.valuegrid import (
    Grid,
    ValueGrid,
    FoodGrid,
    GaussianValueLayer,
    ThermoScape,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_space():
    """Mock agentpy.Space for grid initialization."""
    space = Mock()
    space.dims = (1.0, 1.0)  # 1m x 1m arena
    space.range = (-0.5, 0.5, -0.5, 0.5)  # x0, x1, y0, y1
    return space


@pytest.fixture
def mock_model(mock_space):
    """Mock model with space attribute."""
    model = Mock()
    model.space = mock_space
    return model


@pytest.fixture
def simple_grid():
    """Simple Grid instance for basic tests."""
    return Grid(grid_dims=(10, 10))


@pytest.fixture
def value_grid(mock_model):
    """ValueGrid with mocked model/space."""
    vgrid = ValueGrid(
        grid_dims=(10, 10),
        initial_value=0.0,
        max_value=100.0,
        min_value=0.0,
    )
    vgrid.model = mock_model
    vgrid.match_space(mock_model.space)
    return vgrid


@pytest.fixture
def food_grid(mock_model):
    """FoodGrid with mocked model/space."""
    fgrid = FoodGrid(
        grid_dims=(10, 10),
        initial_value=1e-6,
        max_value=1.0,
    )
    fgrid.model = mock_model
    fgrid.match_space(mock_model.space)
    return fgrid


@pytest.fixture
def mock_odor_source():
    """Mock odor source with gaussian_value method."""
    source = Mock()
    source.get_position = Mock(return_value=(0.0, 0.0))

    # Mock odor with Gaussian value function
    source.odor = Mock()
    source.odor.gaussian_value = Mock(
        side_effect=lambda rel_pos: np.exp(
            -(rel_pos[0] ** 2 + rel_pos[1] ** 2) / (2 * 0.01)
        )
    )
    source.odor.intensity = 10.0
    return source


@pytest.fixture
def gaussian_layer(mock_model, mock_odor_source):
    """GaussianValueLayer with mocked sources."""
    layer = GaussianValueLayer(
        grid_dims=(10, 10),
        sources=[mock_odor_source],
    )
    layer.model = mock_model
    layer.match_space(mock_model.space)
    return layer


@pytest.fixture
def thermo_scape(mock_model):
    """ThermoScape with simple configuration."""
    tscape = ThermoScape(
        grid_dims=(10, 10),
        plate_temp=22,
        spread=0.1,
        thermo_sources=[[0.0, 0.0], [0.5, 0.5]],
        thermo_source_dTemps=[8, -8],  # Hot and cold sources
    )
    tscape.model = mock_model
    tscape.match_space(mock_model.space)
    return tscape


# ============================================================================
# Grid Tests
# ============================================================================


def test_grid_initialization(simple_grid):
    """Test Grid basic initialization."""
    assert simple_grid.grid_dims == (10, 10)
    assert simple_grid.X == 10
    assert simple_grid.Y == 10


def test_grid_X_property(simple_grid):
    """Test Grid X property returns first dimension."""
    assert simple_grid.X == simple_grid.grid_dims[0]


def test_grid_Y_property(simple_grid):
    """Test Grid Y property returns second dimension."""
    assert simple_grid.Y == simple_grid.grid_dims[1]


def test_grid_custom_dimensions():
    """Test Grid with custom dimensions."""
    grid = Grid(grid_dims=(25, 50))
    assert grid.X == 25
    assert grid.Y == 50


# ============================================================================
# ValueGrid Tests
# ============================================================================


def test_valuegrid_initialization(value_grid):
    """Test ValueGrid initialization with default values."""
    assert value_grid.grid_dims == (10, 10)
    assert value_grid.initial_value == 0.0
    assert value_grid.max_value == 100.0
    assert value_grid.min_value == 0.0
    assert value_grid.grid.shape == (10, 10)
    assert np.all(value_grid.grid == 0.0)


def test_valuegrid_sources_default():
    """Test ValueGrid sources default to empty list."""
    vgrid = ValueGrid(grid_dims=(10, 10))
    assert vgrid.sources == []


def test_valuegrid_sources_initialization():
    """Test ValueGrid with provided sources."""
    source1 = Mock()
    source2 = Mock()
    vgrid = ValueGrid(grid_dims=(10, 10), sources=[source1, source2])
    assert len(vgrid.sources) == 2


def test_valuegrid_get_grid_cell(value_grid):
    """Test grid cell calculation from position."""
    # Center position (0, 0) should map to center cell
    cell = value_grid.get_grid_cell((0.0, 0.0))
    assert cell == (5, 5)

    # Bottom-left corner (-0.5, -0.5) should map to (0, 0)
    cell = value_grid.get_grid_cell((-0.5, -0.5))
    assert cell == (0, 0)

    # Top-right corner (0.5, 0.5) should map to (9, 9)
    cell = value_grid.get_grid_cell((0.49, 0.49))
    assert cell == (9, 9)


def test_valuegrid_get_value(value_grid):
    """Test getting value at position."""
    # Set a value directly in grid
    value_grid.grid[5, 5] = 42.0

    # Get value at center position
    value = value_grid.get_value((0.0, 0.0))
    assert value == 42.0


def test_valuegrid_add_value(value_grid):
    """Test adding value to grid cell."""
    # Add value at center
    returned = value_grid.add_value((0.0, 0.0), 25.0)

    # Check grid was updated
    assert value_grid.grid[5, 5] == 25.0
    assert returned == 25.0  # Full value added (no clipping)


def test_valuegrid_add_cell_value_simple(value_grid):
    """Test adding value directly to cell."""
    returned = value_grid.add_cell_value((3, 7), 15.0)
    assert value_grid.grid[3, 7] == 15.0
    assert returned == 15.0


def test_valuegrid_add_cell_value_with_clipping_max(value_grid):
    """Test value addition with max clipping."""
    # Add value that exceeds max
    value_grid.grid[2, 2] = 90.0
    returned = value_grid.add_cell_value((2, 2), 20.0)

    # Max value updates when fixed_max=False, so max becomes 110
    # But grid value should be clipped to updated max
    assert value_grid.grid[2, 2] == 110.0
    assert value_grid.max_value == 110.0
    # Return should be the full value added
    assert returned == 20.0


def test_valuegrid_add_cell_value_with_clipping_min(value_grid):
    """Test value addition with min clipping."""
    # Add negative value below min
    value_grid.grid[4, 4] = 5.0
    returned = value_grid.add_cell_value((4, 4), -10.0)

    # Should clip to min_value (0.0)
    assert value_grid.grid[4, 4] == 0.0
    # Return should be actual change (0 - 5 = -5)
    assert returned == -5.0


def test_valuegrid_add_cell_value_updates_max(value_grid):
    """Test that max_value updates when fixed_max=False."""
    assert value_grid.fixed_max == False
    initial_max = value_grid.max_value

    # Add very large value
    value_grid.add_cell_value((1, 1), 200.0)

    # max_value should update
    assert value_grid.max_value > initial_max
    assert value_grid.max_value == 200.0


def test_valuegrid_add_cell_value_fixed_max(mock_model):
    """Test that max_value doesn't update when fixed_max=True."""
    vgrid = ValueGrid(grid_dims=(10, 10), max_value=100.0, fixed_max=True)
    vgrid.model = mock_model
    vgrid.match_space(mock_model.space)

    initial_max = vgrid.max_value
    vgrid.add_cell_value((1, 1), 200.0)

    # max_value should NOT update
    assert vgrid.max_value == initial_max
    # Value should be clipped
    assert vgrid.grid[1, 1] == 100.0


def test_valuegrid_reset(value_grid):
    """Test reset returns grid to initial_value."""
    # Modify grid
    value_grid.grid[3, 3] = 50.0
    value_grid.grid[7, 2] = 75.0

    # Reset
    value_grid.reset()

    # All values should be initial_value
    assert np.all(value_grid.grid == value_grid.initial_value)


def test_valuegrid_empty_grid(value_grid):
    """Test empty_grid sets all values to zero."""
    # Set some non-zero values
    value_grid.grid[1, 1] = 10.0
    value_grid.grid[5, 5] = 20.0

    # Empty grid
    value_grid.empty_grid()

    # All values should be zero
    assert np.all(value_grid.grid == 0.0)


def test_valuegrid_cel_pos(value_grid):
    """Test cell position calculation."""
    # Center cell (5, 5) - actual position depends on grid cell size
    pos = value_grid.cel_pos(5, 5)
    # Should return a tuple of floats
    assert isinstance(pos, tuple)
    assert len(pos) == 2

    # Cell (0, 0) should be at bottom-left
    pos = value_grid.cel_pos(0, 0)
    assert pos[0] < 0 and pos[1] < 0


def test_valuegrid_cell_vertices(value_grid):
    """Test cell vertices calculation."""
    vertices = value_grid.cell_vertices(5, 5)

    # Should have 4 vertices
    assert vertices.shape == (4, 2)

    # Vertices should form a quadrilateral
    # Check that vertices are distinct
    assert not np.allclose(vertices[0], vertices[1])


def test_valuegrid_generate_grid_vertices(value_grid):
    """Test grid vertices generation."""
    vertices = value_grid.generate_grid_vertices()

    # Should have vertices for all cells
    assert vertices.shape == (value_grid.X, value_grid.Y, 4, 2)


# ============================================================================
# FoodGrid Tests
# ============================================================================


def test_foodgrid_initialization(food_grid):
    """Test FoodGrid initialization with defaults."""
    # unique_id is actually inherited from Object base class, defaults to "Object"
    # unless explicitly set during instantiation
    assert food_grid.color == "green"
    assert food_grid.fixed_max == True
    assert food_grid.initial_value == 1e-6


def test_foodgrid_get_color_min_value(food_grid):
    """Test FoodGrid color at minimum value."""
    color = food_grid.get_color(food_grid.min_value)

    # At min, should be white (255, 255, 255)
    assert np.allclose(color, [255, 255, 255])


def test_foodgrid_get_color_max_value(food_grid):
    """Test FoodGrid color at maximum value."""
    # Mock util.col_range to return green at max
    from larvaworld.lib import util

    color = food_grid.get_color(food_grid.max_value)

    # Should be green (actual RGB depends on util.col_range)
    assert len(color) == 3  # RGB tuple
    assert color[1] > 0  # Green channel should be non-zero


def test_foodgrid_get_color_mid_value(food_grid):
    """Test FoodGrid color at mid-range value."""
    mid_value = (food_grid.min_value + food_grid.max_value) / 2
    color = food_grid.get_color(mid_value)

    # Should be between white and green
    assert len(color) == 3


# ============================================================================
# GaussianValueLayer Tests
# ============================================================================


def test_gaussian_layer_initialization(gaussian_layer):
    """Test GaussianValueLayer initialization."""
    assert gaussian_layer.odorscape == "Gaussian"
    assert len(gaussian_layer.sources) == 1


def test_gaussian_layer_get_value_at_source(gaussian_layer):
    """Test odor value at source position."""
    # At source position (0, 0), value should be maximal
    value = gaussian_layer.get_value((0.0, 0.0))

    # Should be close to 1.0 (Gaussian peak)
    assert value > 0.9


def test_gaussian_layer_get_value_away_from_source(gaussian_layer):
    """Test odor value away from source."""
    # Far from source
    value = gaussian_layer.get_value((0.5, 0.5))

    # Should be much smaller than at source
    assert value < 0.1


def test_gaussian_layer_multiple_sources(mock_model):
    """Test GaussianValueLayer with multiple sources."""
    source1 = Mock()
    source1.get_position = Mock(return_value=(0.0, 0.0))
    source1.odor = Mock()
    source1.odor.gaussian_value = Mock(return_value=0.5)

    source2 = Mock()
    source2.get_position = Mock(return_value=(0.3, 0.3))
    source2.odor = Mock()
    source2.odor.gaussian_value = Mock(return_value=0.3)

    layer = GaussianValueLayer(
        grid_dims=(10, 10),
        sources=[source1, source2],
    )
    layer.model = mock_model
    layer.match_space(mock_model.space)

    # Value should be sum of both sources
    value = layer.get_value((0.1, 0.1))
    assert value == 0.8  # 0.5 + 0.3


def test_gaussian_layer_get_grid(gaussian_layer):
    """Test get_grid returns 2D array."""
    grid = gaussian_layer.get_grid()

    # Should be 2D array matching grid dimensions
    assert grid.shape == gaussian_layer.grid_dims

    # max_value should be updated
    assert gaussian_layer.max_value > 0


# ============================================================================
# ThermoScape Tests
# ============================================================================


def test_thermoscape_initialization(thermo_scape):
    """Test ThermoScape initialization."""
    # unique_id is inherited from Object base class, defaults to "Object"
    assert thermo_scape.plate_temp == 22
    assert thermo_scape.thermo_spread == 0.1
    assert len(thermo_scape.thermo_sources) == 2
    assert len(thermo_scape.thermo_source_dTemps) == 2


def test_thermoscape_sources_dict_format(thermo_scape):
    """Test that thermo sources are stored as dict."""
    # Sources should be dict with string keys
    assert isinstance(thermo_scape.thermo_sources, dict)
    assert "0" in thermo_scape.thermo_sources
    assert "1" in thermo_scape.thermo_sources


def test_thermoscape_generate_thermoscape(thermo_scape):
    """Test thermoscape layer generation."""
    # Should have thermoscape_layers for each source
    assert hasattr(thermo_scape, "thermoscape_layers")
    assert len(thermo_scape.thermoscape_layers) == 2


def test_thermoscape_get_value_at_hot_source(thermo_scape):
    """Test thermal gain at hot source position."""
    # At hot source (0, 0) with dTemp=+8
    gains = thermo_scape.get_value((0.0, 0.0))

    # Should have warm gain > 0, cool gain = 0
    assert "warm" in gains
    assert "cool" in gains
    assert gains["warm"] > 0
    # Cool gain might not be exactly 0 due to second source influence


def test_thermoscape_get_value_at_cold_source(thermo_scape):
    """Test thermal gain at cold source position."""
    # At cold source (0.5, 0.5) with dTemp=-8
    gains = thermo_scape.get_value((0.5, 0.5))

    # Should have cool gain > 0, warm gain = 0
    assert gains["cool"] > 0


def test_thermoscape_get_value_neutral_position(thermo_scape):
    """Test thermal gain at neutral position."""
    # Far from both sources
    gains = thermo_scape.get_value((-0.5, -0.5))

    # Both gains should be small
    assert gains["warm"] >= 0
    assert gains["cool"] >= 0
    assert gains["warm"] + gains["cool"] < 1.0


def test_thermoscape_mismatched_sources_temps():
    """Test ValueError when sources and temps don't match."""
    with pytest.raises(ValueError):
        ThermoScape(
            grid_dims=(10, 10),
            thermo_sources=[[0.0, 0.0], [0.5, 0.5]],
            thermo_source_dTemps=[8],  # Only 1 temp for 2 sources
        )


def test_thermoscape_default_sources():
    """Test ThermoScape with default sources."""
    tscape = ThermoScape(grid_dims=(10, 10))

    # Default should have 4 sources
    assert len(tscape.thermo_sources) == 4
    assert len(tscape.thermo_source_dTemps) == 4


def test_thermoscape_spread_none_defaults_to_01():
    """Test that spread=None defaults to 0.1."""
    tscape = ThermoScape(
        grid_dims=(10, 10),
        spread=None,
        thermo_sources=[[0.0, 0.0]],
        thermo_source_dTemps=[8],
    )
    assert tscape.thermo_spread == 0.1
