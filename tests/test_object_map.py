import unittest

import numpy as np

from conceptgraphs_uav.frame import UAVFrame
from conceptgraphs_uav.graph import ConceptGraphBuilder


def frame(step: int) -> UAVFrame:
    return UAVFrame(
        rgb=np.zeros((32, 32, 3), dtype=np.uint8),
        depth=np.full((32, 32), 8.0, dtype=np.float32),
        position=np.zeros(3, dtype=np.float32),
        quaternion_xyzw=np.array([0, 0, 0, 1], dtype=np.float32),
        step=step,
        metadata={},
    )


class Detection:
    label = "bench"
    confidence = 0.8
    bbox_xyxy = [8, 8, 24, 24]
    phrase = "bench"


class ObjectMapTests(unittest.TestCase):
    def test_repeated_visual_observations_confirm_one_object(self):
        builder = ConceptGraphBuilder(point_stride=2)
        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        for step in range(3):
            builder.add_detection(frame(step), Detection(), visual_embedding=embedding)
        nodes = list(builder.scene_graph.graph.nodes(data=True))
        self.assertEqual(len(nodes), 1)
        _, node = nodes[0]
        self.assertEqual(node["observations"], 3)
        self.assertEqual(node["state"], "confirmed")
        self.assertAlmostEqual(float(np.linalg.norm(node["visual_embedding"])), 1.0, places=5)

    def test_incompatible_visual_features_do_not_merge(self):
        builder = ConceptGraphBuilder(point_stride=2)
        builder.add_detection(frame(0), Detection(), visual_embedding=[1.0, 0.0, 0.0])
        builder.add_detection(frame(1), Detection(), visual_embedding=[0.0, 1.0, 0.0])
        self.assertEqual(builder.scene_graph.graph.number_of_nodes(), 2)

    def test_old_tentative_node_is_pruned(self):
        builder = ConceptGraphBuilder(point_stride=2)
        builder.add_detection(frame(0), Detection(), visual_embedding=[1.0, 0.0, 0.0])
        removed = builder.prune_stale_tentative(current_step=30, max_age=25)
        self.assertEqual(len(removed), 1)
        self.assertEqual(builder.scene_graph.graph.number_of_nodes(), 0)


if __name__ == "__main__":
    unittest.main()
