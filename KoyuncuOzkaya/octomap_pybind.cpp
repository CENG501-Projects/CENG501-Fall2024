#include <pybind11/pybind11.h>
#include <pybind11/stl.h>            // For using std::vector
#include <pybind11/numpy.h>          // For NumPy arrays if needed

#include <octomap/ColorOcTree.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

namespace py = pybind11;

//------------------------------------------------------
// A small struct to hold a colored point (x,y,z,R,G,B).
//------------------------------------------------------
struct ColoredPoint {
    float x, y, z;
    unsigned char r, g, b;
};

//------------------------------------------------------
// A simple class wrapping ColorOcTree for usage in Python
//------------------------------------------------------
class OctoMapWrapper {
public:
    OctoMapWrapper(double resolution)
        : m_tree(resolution)
    {
        // empty
    }

    // Build the octree from a vector of ColoredPoint
    // Mark them as occupied and integrate color
    void buildMap(const std::vector<ColoredPoint>& points) {
        for (auto& cp : points) {
            octomap::point3d pt(cp.x, cp.y, cp.z);
            m_tree.updateNode(pt, true); // occupied
            m_tree.integrateNodeColor(pt.x(), pt.y(), pt.z(), cp.r, cp.g, cp.b);
        }
        m_tree.updateInnerOccupancy();
    }

    // We can prune if we like
    void prune() {
        m_tree.prune();
    }

    // Save to .ot file
    void save(const std::string& filename) {
        m_tree.write(filename);
    }

std::vector<std::tuple<
    std::array<float,3>,  // corner pos
    float,                // occupancy or depth
    std::array<unsigned char,3> // color
>>
getVoxelCorners(float x, float y, float z) 
{
    std::vector<std::tuple<std::array<float,3>, float, std::array<unsigned char,3>>> corners;

    // 1) search for the node that encloses (x,y,z)
    octomap::OcTreeKey key;
    key = m_tree.coordToKey(x, y, z);  // might throw if out of bounds
    octomap::ColorOcTreeNode* node = m_tree.search(key);
    if (!node) {
        // no node found => we can return empty or 8 corners of default
        return corners; // empty => python can check size
    }

    // 2) get the node depth => bounding size
    unsigned depth = m_tree.getTreeDepth(); 
    // Alternatively, you can attempt to guess the node's depth from the key:
    // but OcTree doesn't store the depth in the node directly. 
    // If you want the *exact* node's depth, use e.g.:
    //   unsigned node_depth = m_tree.getTreeDepth(key); 
    // but that function might not exist by default. 
    // We'll do a simpler approach: if we are using a "leaf" node,
    //   we can compute size from the resolution or from the OcTreeKey structure.

    // The *lowest* level node has size = resolution * (1 << (treeDepth - nodeDepth)).
    // For simplicity, let's do something naive:
    double size = m_tree.getNodeSize(depth);

    // 3) bounding box
    // The "center" of that voxel cell is (x,y,z). Actually in ColorOcTree, 
    //   search(x,y,z) returns the node that might not be exactly at center. 
    // Typically, the "center" is m_tree.keyToCoord(key). Let's do that:
    octomap::point3d cpos = m_tree.keyToCoord(key);
    float cx = cpos.x();
    float cy = cpos.y();
    float cz = cpos.z();

    // half-size
    float half = size * 0.5f;

    // 4) define the 8 corners in the usual "trilinear" order
    // e.g. (cx - half, cy - half, cz - half), (cx - half, cy - half, cz + half), ...
    std::vector<octomap::point3d> cornerPts;
    cornerPts.reserve(8);
    cornerPts.push_back(octomap::point3d(cx - half, cy - half, cz - half));
    cornerPts.push_back(octomap::point3d(cx - half, cy - half, cz + half));
    cornerPts.push_back(octomap::point3d(cx - half, cy + half, cz - half));
    cornerPts.push_back(octomap::point3d(cx - half, cy + half, cz + half));
    cornerPts.push_back(octomap::point3d(cx + half, cy - half, cz - half));
    cornerPts.push_back(octomap::point3d(cx + half, cy - half, cz + half));
    cornerPts.push_back(octomap::point3d(cx + half, cy + half, cz - half));
    cornerPts.push_back(octomap::point3d(cx + half, cy + half, cz + half));

    corners.reserve(8);

    for (int i = 0; i < 8; i++) {
        auto cpt = cornerPts[i];
        // search each corner in the tree
        octomap::ColorOcTreeNode* cnode = m_tree.search(cpt);
        float occ_val = 0.0f;
        std::array<unsigned char,3> col_arr = {0,0,0};

        if (cnode) {
            // If you want to interpret occupancy from the log-odds:
            //   float occ = m_tree.isNodeOccupied(cnode) ? 1.0f : 0.0f;
            // Or use the actual occupancy probability:
            float occ_prob = cnode->getOccupancy();
            occ_val = occ_prob;

            // get color
            auto c = cnode->getColor();
            col_arr[0] = c.r;
            col_arr[1] = c.g;
            col_arr[2] = c.b;
        }

        // store ( (cx,cy,cz), occ_val, (r,g,b) )
        std::array<float,3> pos_arr = {
            (float)cpt.x(), (float)cpt.y(), (float)cpt.z()
        };
        corners.push_back(std::make_tuple(pos_arr, occ_val, col_arr));
    }

    return corners;
}

    // Raycast from origin along direction, up to maxRange
    // Return (didHit, distance, r,g,b).
    // If no hit, distance=-1, color=0.
    std::tuple<bool, float, unsigned char, unsigned char, unsigned char>
    castRay(const std::vector<float>& origin,
            const std::vector<float>& direction,
            float maxRange) 
    {
        if (origin.size() != 3 || direction.size() != 3) {
            throw std::runtime_error("origin/direction must have size 3");
        }
        octomap::point3d org(origin[0], origin[1], origin[2]);
        octomap::point3d dir(direction[0], direction[1], direction[2]);
        dir.normalize();

        octomap::point3d end;
        bool hit = m_tree.castRay(org, dir, end, true, double(maxRange));
        if (hit) {
            float dist = (end - org).norm();
            auto* node = m_tree.search(end);
            if (node) {
                auto c = node->getColor();
                return std::make_tuple(true, dist, c.r, c.g, c.b);
            } else {
                // theoretically shouldn't happen if hit is true
                return std::make_tuple(true, dist, 255, 255, 255);
            }
        }
        // no hit
        return std::make_tuple(false, -1.f, 0, 0, 0);
    }

private:
    octomap::ColorOcTree m_tree;
};

//------------------------------------------------------
// Create pybind module
//------------------------------------------------------
PYBIND11_MODULE(octomap_pybind, m) {
    py::class_<ColoredPoint>(m, "ColoredPoint")
        .def(py::init<>())
        .def_readwrite("x", &ColoredPoint::x)
        .def_readwrite("y", &ColoredPoint::y)
        .def_readwrite("z", &ColoredPoint::z)
        .def_readwrite("r", &ColoredPoint::r)
        .def_readwrite("g", &ColoredPoint::g)
        .def_readwrite("b", &ColoredPoint::b);

    py::class_<OctoMapWrapper>(m, "OctoMapWrapper")
        .def(py::init<double>(), py::arg("resolution")=0.01)
        .def("buildMap", &OctoMapWrapper::buildMap)
        .def("prune", &OctoMapWrapper::prune)
        .def("save", &OctoMapWrapper::save)
        .def("castRay", &OctoMapWrapper::castRay)
        .def("getVoxelCorners", &OctoMapWrapper::getVoxelCorners);
}
