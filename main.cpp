#include <iostream>
#include "conv2d_relu.h"
#include "edgemb1.h"
#include "edgemb2.h"
#include "specemb.h"
#include "mixed_freq_attention.h"
#include "decoder.h"
#include "utils.h"
#include "weights_edge_embedding_1.h"
#include "weights_edge_embedding_2.h"
#include "weights_specemb.h"
#include "weights_attention.h"
#include "weights_decoder.h"

int main() {
    try {
        // Load all weights
        std::cout << "Loading weights..." << std::endl;
        load_edge_embedding_weights("./weights/edge_embedding_1/");
        load_edge_embedding_2_weights("./weights/edge_embedding_2/");
        load_specemb_weights("./weights/specemb/");
        load_attention_weights("./weights/attention/");
        load_decoder_weights("./weights/decoder/");

        // Load test image
        std::cout << "Loading test image..." << std::endl;
        Tensor3D input_image = load_image_bin("test_image.bin", 3, 224, 224);

        // Run edge embedding 1
        std::cout << "Running edge embedding 1..." << std::endl;
        Tensor3D edge1_output = edge_embedding_1_forward(input_image);
        std::cout << "Edge1 output shape: [" << edge1_output.size() << "]["
                  << edge1_output[0].size() << "][" << edge1_output[0][0].size() << "]" << std::endl;

        // Run edge embedding 2
        std::cout << "Running edge embedding 2..." << std::endl;
        EdgeEmbedding2Output edge2_output = edge_embedding_2_forward(edge1_output);
        std::cout << "Edge2 feature shape: [" << edge2_output.edge_feature.size() << "]["
                  << edge2_output.edge_feature[0].size() << "][" << edge2_output.edge_feature[0][0].size() << "]" << std::endl;
        std::cout << "Edge2 output shape: [" << edge2_output.edge_output.size() << "]["
                  << edge2_output.edge_output[0].size() << "][" << edge2_output.edge_output[0][0].size() << "]" << std::endl;

        // Run spec embedding
        std::cout << "Running spec embedding..." << std::endl;
        SpecEmbeddingOutput spec_output = spec_embedding_forward(input_image);
        std::cout << "Spec embedding shape: [" << spec_output.embedding.size() << "]["
                  << spec_output.embedding[0].size() << "]" << std::endl;

        // Run mixed frequency attention
        std::cout << "Running mixed frequency attention..." << std::endl;
        Tensor3D attention_output;
        mixed_frequency_attention_forward(
            spec_output.embedding,           // [H*W, C] spectral features
            edge2_output.edge_feature,       // [C, H, W] edge features
            attention_output,                // [C, H, W] output
            14, 14,                         // H, W (assuming 14x14 from resize_nearest)
            256,                            // hidden_size
            1                               // num_heads
        );
        std::cout << "Attention output shape: [" << attention_output.size() << "]["
                  << attention_output[0].size() << "][" << attention_output[0][0].size() << "]" << std::endl;

        // Run decoder
        std::cout << "Running decoder..." << std::endl;
        DecoderParams decoder_params = create_decoder_params();
        Tensor3D final_output = decoder_forward(
            attention_output,                // Refined features from attention
            edge2_output.edge_output,        // Edge features [1, H, W]
            spec_output.feature1,            // spec_feat_0 (first ResNet feature)
            spec_output.feature2,            // spec_feat_1 (second ResNet feature)
            decoder_params
        );
        std::cout << "Final decoder output shape: [" << final_output.size() << "]["
                  << final_output[0].size() << "][" << final_output[0][0].size() << "]" << std::endl;

        std::cout << "All operations completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
