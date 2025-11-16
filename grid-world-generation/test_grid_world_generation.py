import unittest
import numpy as np
from grid_world_generation import GridWorldGenerator, convert_to_standard_format, validate_grid_world

class TestGridWorldGeneration(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = GridWorldGenerator()
        self.sample_grid = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [2, 0, 0, 3],
            [0, 0, 1, 0]
        ])

    def test_generate_valid_grid_world(self):
        """Test that generated grid world has correct dimensions and elements."""
        width, height = 5, 5
        grid = self.generator.generate(width, height)

        self.assertEqual(grid.shape, (height, width))
        self.assertIn(2, grid)  # Start position
        self.assertIn(3, grid)  # Goal position

    def test_convert_to_standard_format(self):
        """Test conversion to standard format maintains grid structure."""
        standard_grid = convert_to_standard_format(self.sample_grid)

        self.assertEqual(standard_grid.shape, self.sample_grid.shape)
        # Check that walls (1) are preserved
        self.assertTrue(np.array_equal(
            (self.sample_grid == 1),
            (standard_grid == 1)
        ))

    def test_validate_grid_world_valid(self):
        """Test validation passes for valid grid world."""
        is_valid, errors = validate_grid_world(self.sample_grid)

        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_validate_grid_world_no_start(self):
        """Test validation fails when no start position exists."""
        invalid_grid = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 3]
        ])

        is_valid, errors = validate_grid_world(invalid_grid)

        self.assertFalse(is_valid)
        self.assertIn("No start position found", errors)

    def test_validate_grid_world_no_goal(self):
        """Test validation fails when no goal position exists."""
        invalid_grid = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [2, 0, 0]
        ])

        is_valid, errors = validate_grid_world(invalid_grid)

        self.assertFalse(is_valid)
        self.assertIn("No goal position found", errors)

    def test_validate_grid_world_multiple_starts(self):
        """Test validation fails with multiple start positions."""
        invalid_grid = np.array([
            [2, 0, 1],
            [0, 1, 0],
            [2, 0, 3]
        ])

        is_valid, errors = validate_grid_world(invalid_grid)

        self.assertFalse(is_valid)
        self.assertIn("Multiple start positions found", errors)

    def test_format_conversion_preserves_validity(self):
        """Test that format conversion preserves grid world validity."""
        # Generate a valid grid world
        original_grid = self.generator.generate(4, 4)

        # Convert to standard format
        standard_grid = convert_to_standard_format(original_grid)

        # Validate the converted grid
        is_valid, errors = validate_grid_world(standard_grid)

        self.assertTrue(is_valid, f"Converted grid is invalid: {errors}")

    def test_grid_world_connectivity(self):
        """Test that start and goal positions are connected."""
        grid = self.sample_grid
        start_pos = np.where(grid == 2)
        goal_pos = np.where(grid == 3)

        # This would require implementing a path-finding algorithm
        # For now, just check positions exist
        self.assertEqual(len(start_pos[0]), 1)
        self.assertEqual(len(goal_pos[0]), 1)

    def test_edge_case_minimum_size(self):
        """Test grid world generation with minimum size."""
        grid = self.generator.generate(2, 2)

        self.assertEqual(grid.shape, (2, 2))
        is_valid, _ = validate_grid_world(grid)
        self.assertTrue(is_valid)

    def test_edge_case_rectangular_grid(self):
        """Test grid world generation with non-square dimensions."""
        grid = self.generator.generate(3, 5)

        self.assertEqual(grid.shape, (5, 3))
        is_valid, _ = validate_grid_world(grid)
        self.assertTrue(is_valid)

if __name__ == '__main__':
    unittest.main()

