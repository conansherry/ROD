#include <iostream>
#include <fstream>
#include "RODCluster.h"

int main(int argc, char* argv[])
{
    std::ifstream fin("tmpfile.txt");
    readsense::Elements elements(674);
    for (int i = 0; i < elements.size(); i++)
    {
        elements[i].id = i;
        elements[i].feature.resize(768);
        for (int j = 0; j < elements[i].feature.size(); j++)
        {
            fin >> elements[i].feature[j];
        }
    }
    readsense::RODCluster rodcluster;
    rodcluster.update(elements);
    readsense::Elements dst;
    rodcluster.identifyClusters(dst);
    for (int i = 0; i < dst.size(); i++)
    {
        std::cout << "id " << dst[i].id << " cluster_id " << dst[i].cluser_id << " coef " << dst[i].cluster_confidence << std::endl;
    }
    /*std::ifstream fin("D:/readsense/tmpFeatures.txt");
    readsense::Elements elements;
    while (true)
    {
        std::string filename;
        fin >> filename;
        if (filename.size() < 5)
            break;
        readsense::Feature feature(256);
        for (int i = 0; i < 256; i++)
        {
            fin >> feature[i];
        }
        readsense::Element element;
        element.id_name = filename;
        element.feature = feature;
        elements.push_back(element);
    }
    readsense::RODCluster rc;
    rc.update(elements);
    readsense::Elements dst;
    rc.identifyClusters(dst);
    std::ofstream fout("D:/readsense/output.txt");
    for (int i = 0; i < dst.size(); i++)
    {
        fout << dst[i].id_name << " " << dst[i].cluser_id << " " << dst[i].cluster_confidence << std::endl;
    }*/
    return 0;
}