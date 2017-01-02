#ifndef RODCLUSTER_H
#define RODCLUSTER_H

#include <string>
#include <vector>
#include <list>

namespace readsense
{

typedef std::vector<float> Feature;
typedef std::pair<int, float> Neighbor;
typedef std::vector<Neighbor> Neighbors;
typedef std::vector<Neighbors> Neighborhood;
typedef std::vector<int> Cluster;
typedef std::vector<Cluster> Clusters;

struct Element
{
    int id;
    std::string id_name;

    int cluser_id;
    float cluster_confidence;

    Feature feature;
};

typedef std::vector<Element> Elements;

class RODCluster
{
public:
    RODCluster();
    virtual ~RODCluster();

    void update(const Elements &src);

    void identifyClusters(Elements& dst);

    std::vector<float> compare(const Elements& templates_, const Element& t);

    float compare(const Element& a, const Element& b);

    Elements templates;

    int kNN;
    float aggression;

    Neighborhood neighborhood;
};

}

#endif