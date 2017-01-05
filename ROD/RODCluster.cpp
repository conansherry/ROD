#include "RODCluster.h"
#include <algorithm>
#include <cassert>
#include <hash_map>
#include <set>
#include <cmath>

namespace readsense
{

template <typename RandomAccessIterator, typename T, typename LessThan>
RandomAccessIterator qLowerBoundHelper(RandomAccessIterator begin, RandomAccessIterator end, const T &value, LessThan lessThan)
{
    RandomAccessIterator middle;
    int n = std::distance(begin, end);
    int half;

    while (n > 0) {
        half = n >> 1;
        middle = std::next(begin, half);
        if (lessThan(*middle, value)) {
            begin = std::next(middle, 1);
            n -= half + 1;
        }
        else {
            n = half;
        }
    }
    return begin;
}

// Compare function used to order neighbors from highest to lowest similarity
static bool compareNeighbors(const Neighbor &a, const Neighbor &b)
{
    if (a.second == b.second)
        return a.first < b.first;
    return a.second > b.second;
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
// Ob(x) in eq. 1, modified to consider 0/1 as ground truth imposter/genuine.
static int indexOf(const Neighbors &neighbors, int i)
{
    for (int j = 0; j<neighbors.size(); j++) {
        //const Neighbor &neighbor = neighbors[j];
        const Neighbor &neighbor = *std::next(neighbors.begin(), j);
        if (neighbor.first == i) {
            if (neighbor.second == 0)
                return neighbors.size() - 1;
            else if (neighbor.second == 1)
                return 0;
            else
                return j;
        }
    }
    return -1;
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
// Corresponds to eq. 1, or D(a,b)
static int asymmetricalROD(const Neighborhood &neighborhood, int a, int b)
{
    int distance = 0;
    for(const Neighbor &neighbor : neighborhood[a]) {
        if (neighbor.first == b) break;
        int index = indexOf(neighborhood[b], neighbor.first);
        distance += (index == -1) ? neighborhood[b].size() : index;
    }
    return distance;
}

// Zhu et al. "A Rank-Order Distance based Clustering Algorithm for Face Tagging", CVPR 2011
// Corresponds to eq. 2/4, or D-R(a,b)
float normalizedROD(const Neighborhood &neighborhood, int a, int b)
{
    int indexA = indexOf(neighborhood[b], a);
    int indexB = indexOf(neighborhood[a], b);

    // Default behaviors
    if ((indexA == -1) || (indexB == -1)) return std::numeric_limits<float>::max();
    if ((*std::next(neighborhood[b].begin(), indexA)).second == 1 || (*std::next(neighborhood[a].begin(), indexB)).second == 1) return 0;
    if ((*std::next(neighborhood[b].begin(), indexA)).second == 0 || (*std::next(neighborhood[a].begin(), indexB)).second == 0) return std::numeric_limits<float>::max();

    int distanceA = asymmetricalROD(neighborhood, a, b);
    int distanceB = asymmetricalROD(neighborhood, b, a);
    return 1.f * (distanceA + distanceB) / std::min(indexA + 1, indexB + 1);
}

// Rank-order clustering on a pre-computed k-NN graph
Clusters ClusterGraph(Neighborhood neighborhood, float aggressiveness)
{
    const int cutoff = neighborhood.front().size();
    const float threshold = 3 * cutoff / 4 * aggressiveness / 5;

    // Initialize clusters
    Clusters clusters(neighborhood.size());
    for (int i = 0; i<neighborhood.size(); i++)
        clusters[i].push_back(i);

    bool done = false;
    while (!done) {
        // nextClusterIds[i] = j means that cluster i is set to merge into cluster j
        std::vector<int> nextClusterIDs(neighborhood.size());
        for (int i = 0; i<neighborhood.size(); i++) nextClusterIDs[i] = i;

        // For each cluster
        for (int clusterID = 0; clusterID<neighborhood.size(); clusterID++) {
            const Neighbors &neighbors = neighborhood[clusterID];
            int nextClusterID = nextClusterIDs[clusterID];

            // Check its neighbors
            for(const Neighbor &neighbor : neighbors) {
                int neighborID = neighbor.first;
                int nextNeighborID = nextClusterIDs[neighborID];

                // Don't bother if they have already merged
                if (nextNeighborID == nextClusterID) continue;

                // Flag for merge if similar enough
                if (normalizedROD(neighborhood, clusterID, neighborID) < threshold) {
                    if (nextClusterID < nextNeighborID)
                        nextClusterIDs[neighborID] = nextClusterID;
                    else
                        nextClusterIDs[clusterID] = nextNeighborID;
                }
            }
        }

        // Transitive merge
        for (int i = 0; i<neighborhood.size(); i++) {
            int nextClusterID = i;
            while (nextClusterID != nextClusterIDs[nextClusterID]) {
                assert(nextClusterIDs[nextClusterID] < nextClusterID);
                nextClusterID = nextClusterIDs[nextClusterID];
            }
            nextClusterIDs[i] = nextClusterID;
        }

        // Construct new clusters
        std::hash_map<int, int> clusterIDLUT;
        std::set<int> tmpUniqueS(nextClusterIDs.begin(), nextClusterIDs.end());
        std::vector<int> allClusterIDs(tmpUniqueS.begin(), tmpUniqueS.end());
        for (int i = 0; i < neighborhood.size(); i++)
        {
            clusterIDLUT[i] = find(allClusterIDs.begin(), allClusterIDs.end(), nextClusterIDs[i]) - allClusterIDs.begin();
        }

        Clusters newClusters(allClusterIDs.size());
        Neighborhood newNeighborhood(allClusterIDs.size());

        for (int i = 0; i<neighborhood.size(); i++) {
            int newID = clusterIDLUT[i];
            newClusters[newID].insert(newClusters[newID].end(), clusters[i].begin(), clusters[i].end());
            newNeighborhood[newID].insert(newNeighborhood[newID].end(), neighborhood[i].begin(), neighborhood[i].end());
        }

        // Update indices and trim
        for (int i = 0; i<newNeighborhood.size(); i++) {
            Neighbors &neighbors = newNeighborhood[i];
            int size = std::min((int)neighbors.size(), cutoff);

            std::partial_sort(neighbors.begin(), neighbors.begin() + size, neighbors.end(), compareNeighbors);

            for (int j = 0; j<size; j++)
                (*std::next(neighbors.begin(), j)).first = clusterIDLUT[j];
            neighbors = Neighbors(neighbors.begin(), std::next(neighbors.begin(), cutoff));
            //neighbors = neighbors.mid(0, cutoff);
        }

        // Update results
        done = true; //(newClusters.size() >= clusters.size());
        clusters = newClusters;
        neighborhood = newNeighborhood;
    }

    return clusters;
}

RODCluster::RODCluster()
{
    kNN = 4;
    aggression = 100;
}

RODCluster::~RODCluster()
{
}

void RODCluster::setParas(int _kNN, float _aggression)
{
    kNN = _kNN;
    aggression = _aggression;
}

void RODCluster::update(const Elements & _src)
{
    Elements src = _src;
    for (Element &element : src)
    {
        double fNormFea = 0;
        for (int i = 0; i < element.feature.size(); i++)
        {
            fNormFea += element.feature[i] * element.feature[i];
        }
        fNormFea = std::sqrtf(fNormFea);
        for (int i = 0; i < element.feature.size(); i++)
        {
            element.feature[i] /= (fNormFea + 0.000001f);
        }
    }
    // update current graph
    for(const Element &t : src) {
        std::vector<float> scores = compare(templates, t);
        // attempt to udpate each existing point's (sorted) k-NN list with these results.
        Neighbors currentN;
        for (int i = 0; i < scores.size(); i++) {
            currentN.push_back(Neighbor(i, scores[i]));
            Neighbors target = neighborhood[i];

            // should we insert the new neighbor into the current target's list?
            if (target.size() < kNN || scores[i] > target.back().second) {
                // insert into the sorted nearest neighbor list
                Neighbor temp(scores.size(), scores[i]);
                Neighbors::iterator res = qLowerBoundHelper(target.begin(), target.end(), temp, compareNeighbors);
                target.insert(res, temp);

                if (target.size() > kNN)
                    target.pop_back();

                neighborhood[i] = target;
            }
        }

        // add a new row, consisting of the top neighbors of the newest point
        int actuallyKeep = std::min(kNN, (int)currentN.size());
        std::partial_sort(currentN.begin(), currentN.begin() + actuallyKeep, currentN.end(), compareNeighbors);

        Neighbors selected = Neighbors(currentN.begin(), std::next(currentN.begin(), actuallyKeep));//currentN.mid(0, actuallyKeep);
        neighborhood.push_back(selected);
        templates.push_back(t);
    }
}

void RODCluster::identifyClusters(Elements & dst)
{
    Clusters clusters = ClusterGraph(neighborhood, aggression);
    for (int i = 0; i < clusters.size(); i++) {
        // Calculate the centroid of each cluster
        Feature center(templates[0].feature.size(), 0);
        for(int t : clusters[i]) {
            for (int j = 0; j < center.size(); j++)
                center[j] += templates[t].feature[j];
        }
        for (int j = 0; j < center.size(); j++)
            center[j] /= clusters[i].size();

        // Calculate the Euclidean distance from the center to use as the cluster confidence
        for(int t : clusters[i]) {
            templates[t].cluser_id = i;
            Feature p(center.size(), 0);
            for (int j = 0; j < p.size(); j++)
            {
                p[j] = std::pow(templates[t].feature[j] - center[j], 2);
            }
            double sum = 0;
            for (int j = 0; j < p.size(); j++)
            {
                sum += p[j];
            }
            templates[t].cluster_confidence = std::sqrt(sum);
        }
    }
    dst.insert(dst.end(), templates.begin(), templates.end());
}

std::vector<float> RODCluster::compare(const Elements & templates_, const Element & t)
{
    std::vector<float> scores;
    scores.reserve(templates_.size());
    for (const Element &target : templates_)
        scores.push_back(compare(target, t));
    return scores;
}

float RODCluster::compare(const Element & a, const Element & b)
{
    assert(a.feature.size() == b.feature.size());
    double sum = 0;
    for (int i = 0; i < a.feature.size(); i++)
    {
        sum += std::pow(a.feature[i] - b.feature[i], 2);
    }
    sum = std::sqrt(sum);
    return -std::log(sum + 1);
}

}