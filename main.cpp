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
	std::string img_path = "../../data/images/03.jpg";
	cv::Mat oriImage = cv::imread(img_path);
	if (oriImage.empty()) {
		printf("invalid img!\n");
		exit(EXIT_FAILURE);
	}

	// model load and forward
	cv::cvtColor(oriImage, oriImage, cv::COLOR_RGB2BGR);
	HandTailor* ptrHandTailor = new HandTailor(model_path);
	printf("load model successful!\n");
	std::vector<cv::Point2f> coords;
	cv::Vec<float, 48> so3_vec;
	cv::Vec<float, 10> beta_vec;
	cv::Vec<float, 3> joint_root_vec;
	cv::Mat bone;
	ptrHandTailor->get_param(coords, so3_vec, beta_vec, joint_root_vec,bone, oriImage);

	//MANO model
	MANO right_mano;
	right_mano.load_model("../../data/mpi_model/hand/mano_right.dat");
	MatrixXf mano_r_v, mano_r_joints, mano_r_skel;
	MatrixXi mano_r_f;

	VectorXf mano_r_shape = Eigen::Map<Eigen::VectorXf>(beta_vec.val, 10);
	VectorXf mano_r_pose(52);
	mano_r_pose[0] = 1.f;
	mano_r_pose.middleRows(1, 3).setZero();
	mano_r_pose.bottomRows(48) = Eigen::Map<Eigen::VectorXf>(so3_vec.val, 48);
	right_mano.deform(mano_r_v, mano_r_joints, mano_r_pose, mano_r_shape, true, false);
	right_mano.getFaces(mano_r_f);

	//MANO result processing...
	MatrixXf Joints3D, Mesh_verts;
	ptrHandTailor->convertJointsandVerts(Joints3D, Mesh_verts, mano_r_joints, mano_r_v);  //size(3,21)
	float bone_length = (Joints3D.col(0) - Joints3D.col(9)).norm();
	Mesh_verts = Mesh_verts / bone_length;
	Joints3D = Joints3D / bone_length;

	//obtain scale
	float scale;
	ptrHandTailor->obtain_scale(scale, Joints3D, coords);
	cout << "scale"<<scale << endl;

	//trans verts
	MatrixXf Verts;
	Verts = Mesh_verts.array() * scale;
	Verts = Verts.colwise() + Vector3f(coords[9].x, coords[9].y, 0);
	//trans joints
	MatrixXf Joints;
	Joints = Joints3D.array() * scale;
	Joints = Joints.colwise() + Vector3f(coords[9].x, coords[9].y, 0);

	//visualize the result
	MatrixXf np_Vert, np_joints;
	load_nptxt(np_Vert, R"(C:\Users\zbh\Desktop\test\vert_trans_noopt.txt)");
	load_nptxt(np_joints, R"(C:\Users\zbh\Desktop\test\joints_trans_noopt.txt)");
	gui.add_coord_frame();
	gui.add_mesh(Verts, mano_r_f);
	//gui.add_points(np_Vert.transpose());
	//gui.add_mesh(np_Vert.transpose(), mano_r_f, GREEN, 0.1);
	gui.add_primitives(VCL_PRIMITIVE::sphere, Joints, RED, Vector1f(0.01));
	//gui.add_primitives(VCL_PRIMITIVE::sphere, np_joints.transpose(), GREEN, Vector1f(0.01));
	gui.launch();
	
}
