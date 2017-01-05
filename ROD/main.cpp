#include <iostream>
#include <fstream>
#include "RODCluster.h"
#include <hash_map>
#include <set>

int main(int argc, char* argv[])
{
    //std::ifstream fin("tmpfile.txt");
    //readsense::Elements elements(674);
    //for (int i = 0; i < elements.size(); i++)
    //{
    //    elements[i].id = i;
    //    elements[i].feature.resize(768);
    //    for (int j = 0; j < elements[i].feature.size(); j++)
    //    {
    //        fin >> elements[i].feature[j];
    //    }
    //}
    //readsense::RODCluster rodcluster;
    //rodcluster.update(elements);
    //readsense::Elements dst;
    //rodcluster.identifyClusters(dst);
    //for (int i = 0; i < dst.size(); i++)
    //{
    //    std::cout << "id " << dst[i].id << " cluster_id " << dst[i].cluser_id << " coef " << dst[i].cluster_confidence << std::endl;
    //}

    std::ifstream fin("test_feature_save_p2_gray.txt");
    readsense::Elements elements(1827);
    for (int i = 0; i < elements.size(); i++)
    {
        std::string name;
        fin >> name;
        elements[i].id = i;
        elements[i].id_name = name;
        int x, y, w, h;
        fin >> x >> y >> w >> h;
        elements[i].feature.resize(256);
        for (int j = 0; j < 256; j++)
        {
            float v;
            fin >> v;
            elements[i].feature[j] = v;
        }
    }
    readsense::RODCluster rodcluster;
    rodcluster.update(elements);
    readsense::Elements dst;
    rodcluster.identifyClusters(dst);
    std::hash_map<int, std::set<std::string> > results;
    for (int i = 0; i < dst.size(); i++)
    {
        std::cout << "id_name " << dst[i].id_name << " cluster_id " << dst[i].cluser_id << " coef " << dst[i].cluster_confidence << std::endl;
        results[dst[i].cluser_id].insert(dst[i].id_name);
    }
    for (auto i : results)
    {
        if (i.second.size() > 1)
        {
            std::cout << "Error cluster_id = " << i.first << std::endl;
        }
    }
    return 0;
}