// ========================= oookkkkk
#include <tensorflow/c/c_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <csignal> // just for the g_signal_caught

// Dummy deallocator function
void NoOpDeallocator(void* data, size_t a, void* b) {}

// TODO Signal handler function
// but still that would not handle the case if the program crash....
// and the pointers should be global....
volatile sig_atomic_t g_signal_caught = 0;
void signal_handler(int signal) {
    if (signal == SIGINT) {
        std::cout << "SIGINT (Ctrl-C) caught. Cleaning up resources..." << std::endl;

        g_signal_caught = 1;
    }
}
// end of to do.....

// Function to read the model file
std::vector<char> read_file(const std::string& file_path) {
	std::ifstream file(file_path, std::ios::binary);
	if (!file) {
	    throw std::runtime_error("Could not open file: " + file_path);
	}
	
	std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

	// // to debug
	// // Read the file into a vector<char>
    	// // Print out the contents of the vector
    	// std::cout << "Model Data Size: " << buffer.size() << std::endl;
    	// std::cout << "First 100 bytes: ";
    	// for (int i = 0; i < std::min<int>(100, buffer.size()); ++i) {
    	//     std::cout << static_cast<int>(buffer[i]) << " ";
    	// }
    	// std::cout << std::endl;
    	// // end debug



	return buffer;
}

// Function to preprocess the image
std::vector<float> preprocess_image(const std::string& image_path) {
	cv::Mat img = cv::imread(image_path);
	if (img.empty()) {
	    throw std::runtime_error("Could not read image: " + image_path);
	}
	
	cv::resize(img, img, cv::Size(150, 150)); // Resize image to 150x150
	img.convertTo(img, CV_32F); // Convert image to float
	img = img / 255.0; // Normalize pixel values
	
	std::vector<float> img_data;
	img_data.assign((float*)img.datastart, (float*)img.dataend);
	return img_data;
}

int main() {
	// Initialize TensorFlow
	TF_Status* status = TF_NewStatus();
	TF_Graph* graph = TF_NewGraph();
	TF_SessionOptions* options = TF_NewSessionOptions();
	TF_Buffer* run_options = NULL;
	TF_Session* session = TF_NewSession(graph, options, status);

	printf("step 0");
	
	if (TF_GetCode(status) != TF_OK) {
	    	std::cerr << "Error: Unable to create session " << TF_Message(status) << std::endl;
		return -1;
	}
	
	// Load model
	const std::string model_path = "../saved_model_pb/saved_model.pb";
	std::vector<char> model_data = read_file(model_path);

	
	TF_Buffer graph_def_buffer = { model_data.data(), model_data.size(), nullptr };
	TF_ImportGraphDefOptions* graph_def_options = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(graph, &graph_def_buffer, graph_def_options, status);
	if (TF_GetCode(status) != TF_OK) {
	    	std::cerr << "Error: Unable to import graph " << TF_Message(status) << std::endl;
	    	return -1;
	}
	
	TF_DeleteImportGraphDefOptions(graph_def_options);

	
	// // List all operations in the graph to verify their name,
	// // and find the input_op and output_op names.
	// size_t pos = 0;
	// TF_Operation* oper;
	// printf("Operations in the graph:\n");
	// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
	// 	printf("%s\n", TF_OperationName(oper));
	// }
	
	for(short i = 1; i < 10; i++){
		
		// Load and preprocess image
		const std::string image_path = "../testImage/" + std::to_string(i) + ".JPG";
		std::vector<float> img_data = preprocess_image(image_path);
		
		// check the size of the image(then the input_tensor must match it)
		// std::cout << "Size of img_data: " << img_data.size() << std::endl;
		
		// Define input tensor
		// !!! "x" is the first line of the output from the "Operations in the graph:\n"
		TF_Output input_op = { TF_GraphOperationByName(graph, "x"), 0 }; // Replace with your input tensor name
		const int64_t dim[4] = { 1, 150, 150, 3 }; // Batch size of 1, 150x150 image, 3 channels (RGB) => 1 * 150 * 150 * 3 = 67500
		TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, dim, 4, img_data.data(), img_data.size() * sizeof(float), NoOpDeallocator, nullptr);
		
		
		// Define output tensor, 
		// !!! "Identity" is the last line of the output from the "Operations in the graph:\n"
		TF_Output output_op = { TF_GraphOperationByName(graph, "Identity"), 0 }; // Replace with your output tensor name
		TF_Tensor* output_tensor = nullptr;
		
		// Run session
		TF_SessionRun(session, nullptr,
		              &input_op, &input_tensor, 1,
		              &output_op, &output_tensor, 1,
		              nullptr, 0, nullptr, status);
		if (TF_GetCode(status) != TF_OK) {
		    	std::cerr << "Error: Unable to run session " << TF_Message(status) << std::endl;
		    	return -1;
		}

		// Interpret output
		float* predictions = static_cast<float*>(TF_TensorData(output_tensor));
		int predicted_index = std::distance(predictions, std::max_element(predictions, predictions + 14));
		std::vector<std::string> class_to_view = {
		    	"eo_front", "eo_front_smile", "eo_front_wide_smile", "eo_left",
		    	"eo_left_smile", "eo_right", "eo_right_smile", "io_front_occlusion",
		    	"io_front_occlusion_upwork", "io_left_occlusion", "io_lower_arch",
		    	"io_right_occlusion", "io_upper_arch", "o_xray"
		};
		std::cout << "Predicted view: " << class_to_view[predicted_index] << std::endl;
	
		TF_DeleteTensor(input_tensor);
		TF_DeleteTensor(output_tensor);
	}
	
	// Clean up
	TF_DeleteSession(session, status);
	TF_DeleteSessionOptions(options);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(status);
	
	return 0;
}



// =========== try to implement with smart pointer but does not work ======================
// #include <tensorflow/c/c_api.h>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <stdexcept>
// #include <opencv2/opencv.hpp>
// #include <memory> // Include the <memory> header for smart pointers
// 
// // Define a custom deleter for TF_Buffer
// void TF_BufferDeleter(void* data, size_t length) {
//     if (data != nullptr) {
//         TF_DeleteBuffer(static_cast<TF_Buffer*>(data));
//     }
// }
// 
// // Define a custom deleter for TF_SessionOptions
// void TF_SessionOptionsDeleter(TF_SessionOptions* options) {
//     if (options != nullptr) {
//         TF_DeleteSessionOptions(options);
//     }
// }
// 
// // Define a custom deleter for TF_Graph
// void TF_GraphDeleter(TF_Graph* graph) {
//     if (graph != nullptr) {
//         TF_DeleteGraph(graph);
//     }
// }
// 
// // Define a custom deleter for TF_Session
// void TF_SessionDeleter(TF_Session* session) {
//     if (session != nullptr) {
//         TF_Status* status = TF_NewStatus();
//         TF_CloseSession(session, status);
//         TF_DeleteSession(session, status);
//         TF_DeleteStatus(status);
//     }
// }
// 
// // Dummy deallocator function
// void NoOpDeallocator(void* data, size_t a, void* b) {}
// 
// // Function to read the model file
// std::vector<char> read_file(const std::string& file_path) {
// 	std::ifstream file(file_path, std::ios::binary);
// 	if (!file) {
// 	    throw std::runtime_error("Could not open file: " + file_path);
// 	}
// 	
// 	std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
// 
// 	return buffer;
// }
// 
// // Function to preprocess the image
// std::vector<float> preprocess_image(const std::string& image_path) {
// 	cv::Mat img = cv::imread(image_path);
// 	if (img.empty()) {
// 	    throw std::runtime_error("Could not read image: " + image_path);
// 	}
// 	
// 	cv::resize(img, img, cv::Size(150, 150)); // Resize image to 150x150
// 	img.convertTo(img, CV_32F); // Convert image to float
// 	img = img / 255.0; // Normalize pixel values
// 	
// 	std::vector<float> img_data;
// 	img_data.assign((float*)img.datastart, (float*)img.dataend);
// 	return img_data;
// }
// 
// int main() {
// 	// Initialize TensorFlow
//     	TF_Status* status = TF_NewStatus();
//     	std::unique_ptr<TF_Graph, decltype(&TF_GraphDeleter)> graph(TF_NewGraph(), TF_GraphDeleter);
//     	std::unique_ptr<TF_SessionOptions, decltype(&TF_SessionOptionsDeleter)> options(TF_NewSessionOptions(), TF_SessionOptionsDeleter);
//     	std::unique_ptr<TF_Session, decltype(&TF_SessionDeleter)> session(TF_NewSession(graph.get(), options.get(), status), TF_SessionDeleter);
// 
// 	// TF_Status* status = TF_NewStatus();
// 	// TF_Graph* graph = TF_NewGraph();
// 	// TF_SessionOptions* options = TF_NewSessionOptions();
// 	// TF_Buffer* run_options = NULL;
// 	// TF_Session* session = TF_NewSession(graph, options, status);
// 	
// 	if (TF_GetCode(status) != TF_OK) {
// 	    	std::cerr << "Error: Unable to create session " << TF_Message(status) << std::endl;
// 		return -1;
// 	}
// 	
// 	// Load model
// 	const std::string model_path = "../saved_model_pb/saved_model.pb";
// 	std::vector<char> model_data = read_file(model_path);
// 
// 	
// 	TF_Buffer* graph_def_buffer = TF_NewBuffer();
//     	graph_def_buffer->data = model_data.data();
//     	graph_def_buffer->length = model_data.size();
//     	graph_def_buffer->data_deallocator = TF_BufferDeleter;
// 
//     	TF_ImportGraphDefOptions* graph_def_options = TF_NewImportGraphDefOptions();
//     	TF_GraphImportGraphDef(graph.get(), graph_def_buffer, graph_def_options, status);
//     	TF_DeleteImportGraphDefOptions(graph_def_options);
// 
//     	if (TF_GetCode(status) != TF_OK) {
//         	std::cerr << "Error: Unable to import graph " << TF_Message(status) << std::endl;
//         	return -1;
//     	}
// 		
// 	// // List all operations in the graph to verify their name,
// 	// // and find the input_op and output_op names.
// 	// size_t pos = 0;
// 	// TF_Operation* oper;
// 	// printf("Operations in the graph:\n");
// 	// while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
// 	// 	printf("%s\n", TF_OperationName(oper));
// 	// }
// 	
// 	for(short i = 1; i < 10; i++){
// 		
// 		// Load and preprocess image
// 		const std::string image_path = "../testImage/" + std::to_string(i) + ".JPG";
// 		std::vector<float> img_data = preprocess_image(image_path);
// 		
// 		// check the size of the image(then the input_tensor must match it)
// 		// std::cout << "Size of img_data: " << img_data.size() << std::endl;
// 		
// 		// Define input tensor
// 		// !!! "x" is the first line of the output from the "Operations in the graph:\n"
// 		TF_Output input_op = { TF_GraphOperationByName(graph.get(), "x"), 0 }; // Replace with your input tensor name
// 		const int64_t dim[4] = { 1, 150, 150, 3 }; // Batch size of 1, 150x150 image, 3 channels (RGB) => 1 * 150 * 150 * 3 = 67500
// 		std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> input_tensor(TF_NewTensor(TF_FLOAT, dim, 4, img_data.data(), img_data.size() * sizeof(float), nullptr, nullptr), TF_DeleteTensor);
// 		
// 		// Define output tensor, 
// 		// !!! "Identity" is the last line of the output from the "Operations in the graph:\n"
// 		TF_Output output_op = { TF_GraphOperationByName(graph.get(), "Identity"), 0 }; // Replace with your output tensor name
// 		
// 		TF_Tensor* input_tensor_ptr = input_tensor.get();
// 		TF_Tensor* output_tensor_ptr = nullptr;
// 
// 		// Run session
// 		TF_SessionRun(session.get(), nullptr,
// 		              &input_op, &input_tensor_ptr, 1,
// 		              &output_op, &output_tensor_ptr, 1,
// 		              nullptr, 0, nullptr, status);
// 		if (TF_GetCode(status) != TF_OK) {
// 		    	std::cerr << "Error: Unable to run session " << TF_Message(status) << std::endl;
// 		    	return -1;
// 		}
// 
// 		// Interpret output
// 		float* predictions = static_cast<float*>(TF_TensorData(output_tensor_ptr));
// 		int predicted_index = std::distance(predictions, std::max_element(predictions, predictions + 14));
// 		std::vector<std::string> class_to_view = {
// 		    	"eo_front", "eo_front_smile", "eo_front_wide_smile", "eo_left",
// 		    	"eo_left_smile", "eo_right", "eo_right_smile", "io_front_occlusion",
// 		    	"io_front_occlusion_upwork", "io_left_occlusion", "io_lower_arch",
// 		    	"io_right_occlusion", "io_upper_arch", "o_xray"
// 		};
// 		std::cout << "Predicted view: " << class_to_view[predicted_index] << std::endl;
// 	
// 		// TF_DeleteTensor(output_tensor_ptr);
// 	}
// 	
// 	// Clean up
// 	TF_DeleteStatus(status);
// 	
// 	return 0;
// }



