#include <vector>
#include <Eigen/Eigen> 
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include "HandTailor.hpp"
#include "../../CommonUtils/Models/MpiModel.h"
#include "../../CommonUtils/utils_func.hpp"
#include "../../CommonUtils/Viewer/Viewer_gui.h"

#ifdef _DEBUG
#pragma comment(lib, "opencv_world453d.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "nvrtc.lib")
#pragma comment(lib, "nvinfer.lib")
#pragma comment(lib, "nvparsers.lib")
#pragma comment(lib, "nvonnxparser.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "c10.lib")

#pragma comment(lib, "glfw3dll.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "OpenMeshCored.lib")
#pragma comment(lib, "OpenMeshToolsd.lib")

#pragma comment(lib, "ceresd.lib")
#pragma comment(lib, "../../libs/CommonUtilsd.lib")
#else
#pragma comment(lib, "opencv_world453.lib")
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "nvrtc.lib")
#pragma comment(lib, "nvinfer.lib")
#pragma comment(lib, "nvparsers.lib")
#pragma comment(lib, "nvonnxparser.lib")
#pragma comment(lib, "torch_cpu.lib")
#pragma comment(lib, "torch_cuda.lib")
#pragma comment(lib, "c10.lib")

#pragma comment(lib, "glfw3dll.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "OpenMeshCore.lib")
#pragma comment(lib, "OpenMeshTools.lib")

#pragma comment(lib, "ceres.lib")
#pragma comment(lib, "../../libs/CommonUtils.lib")
#endif

using namespace Eigen;
using namespace std;
using namespace cv;

void main(int argc, char** argv)
{
	Viewer_gui gui;

	const std::string	model_path = "../../data/net_weights/hand/handtailor/model_new.pt";
	std::string img_path = "../../data/images/02.jpg";
	cv::Mat oriImage = cv::imread(img_path);
	if (oriImage.empty()) {
		printf("invalid img!\n");
		exit(EXIT_FAILURE);
	}

	cv::cvtColor(oriImage, oriImage, cv::COLOR_RGB2BGR);
	HandTailor* ptrHandTailor = new HandTailor(model_path);
	printf("load model successful!\n");

	std::vector<cv::Point2f> coords;
	cv::Vec<float, 48> so3_vec;
	cv::Vec<float, 10> beta_vec;
	cv::Mat joint_root, bone;
	ptrHandTailor->get_param(coords, so3_vec, beta_vec,joint_root,bone, oriImage);
	

	//// result test
	//std::cout << "joints" << coords << "\n" << std::endl;
	/*std::cout << "theta" << so3_vec << "\n" << std::endl;
	std::cout << "beta" << beta_vec << "\n" << std::endl;*/
	
	MatrixXf np_Vert;
	load_nptxt(np_Vert, R"(F:\HandTailor\vert_1008.txt)");

	MatrixXf joints3D;
	load_nptxt(joints3D, "../../data/test_file/joints3D.txt");
	float scale;
	ptrHandTailor->obtain_scale(scale, joints3D.transpose(), coords);

	cout << "joint_3D" << joints3D << "\n" << endl;
	cout << "scale" << scale << "\n" << endl;

	float error;
	error = ptrHandTailor->obtain_error(so3_vec, beta_vec, joints3D.transpose(), coords, coords[9], scale);
	cout << "error" << error << endl;
	////MANO model
	//MANO right_mano;
	//right_mano.load_model("../../data/mpi_model/hand/mano_right.dat");
	//MatrixXf mano_r_v, mano_r_joints, mano_r_skel;
	//MatrixXi mano_r_f;

	//VectorXf mano_r_shape = Eigen::Map<Eigen::VectorXf>(beta_vec.val, 10);

	//VectorXf mano_r_pose(52); // mano_r_pose = VectorXf::Random(52) * 2;
	//mano_r_pose[0] = 1.f;
	//mano_r_pose.middleRows(1, 3).setZero();
	//mano_r_pose.bottomRows(48) = Eigen::Map<Eigen::VectorXf>(so3_vec.val, 48);

	//right_mano.deform(mano_r_v, mano_r_joints, mano_r_pose, mano_r_shape, true, false);
	//right_mano.getFaces(mano_r_f);
	//gui.add_coord_frame();
	//gui.add_mesh(mano_r_v*4, mano_r_f);
	//gui.add_points(np_Vert.transpose());
	//gui.launch();

	//ptrHandTailor->draw(oriImage,coords);
	//cv::imshow("image", oriImage);
	//cv::waitKey(-1);
	//printf("forward end!\n");
	

	
}
