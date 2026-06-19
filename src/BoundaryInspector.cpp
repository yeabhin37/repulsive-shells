#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/filesystem.hpp>

#include <goast/Core.h>
#include <OpenMesh/Core/IO/MeshIO.hh>

#include "BoundaryUtils.h"

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 4)
    {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0]
                  << " input_mesh.ply [output_folder] [file_prefix]\n";
        return 1;
    }

    const std::string inputFile = argv[1];

    std::string outputFolder;
    std::string filePrefix = boost::filesystem::path(inputFile).stem().string();
    bool saveFiles = false;

    if (argc >= 3)
    {
        outputFolder = argv[2];

        if (!outputFolder.empty() && outputFolder.back() != '/')
            outputFolder += "/";

        boost::filesystem::create_directories(outputFolder);
        saveFiles = true;
    }

    if (argc == 4)
    {
        filePrefix = argv[3];
    }

    try
    {
        TriMesh mesh;

        if (!OpenMesh::IO::read_mesh(mesh, inputFile))
            throw std::runtime_error("Failed to read mesh: " + inputFile);

        auto boundaryData = BoundaryUtils::extractBoundaryData(mesh);

        BoundaryUtils::printBoundaryInfo(
            boundaryData,
            std::cout,
            inputFile);

        if (saveFiles)
        {
            const std::string outputPrefix = outputFolder;

            BoundaryUtils::writeBoundaryDebugFiles(
                mesh,
                boundaryData,
                outputPrefix,
                filePrefix);

            std::cout << " .. [BoundaryInspector] Saved boundary files to: "
                      << outputFolder << std::endl;

            std::cout << " .. [BoundaryInspector] prefix: "
                      << filePrefix << std::endl;

            std::cout << " .. [BoundaryInspector] "
                      << filePrefix << "_boundary_edges.obj" << std::endl;

            std::cout << " .. [BoundaryInspector] "
                      << filePrefix << "_boundary_edges.csv" << std::endl;

            std::cout << " .. [BoundaryInspector] "
                      << filePrefix << "_boundary_vertices.csv" << std::endl;

            std::cout << " .. [BoundaryInspector] "
                      << filePrefix << "_boundary_marked.ply" << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[BoundaryInspector] Error: "
                  << e.what() << std::endl;
        return 1;
    }

    return 0;
}