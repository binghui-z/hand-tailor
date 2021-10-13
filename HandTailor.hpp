#ifndef __HAND_TAILOR_H
#define __HAND_TAILOR_H
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string.h>
#include <math.h>

#include "../../CommonUtils/Loaders/dnn_loader.hpp"
#include "../../CommonUtils/Loaders/mat_loader.hpp"

using namespace Eigen;
using namespace std;
using namespace cv;

class HandTailor
{
public:
	HandTailor() { m_ptrBone = nullptr; };
	// width_height_ratio: width / height
	HandTailor(const std::string& model_filename)
	{
		std::filesystem::path p(model_filename);
		if (p.extension().string() == ".pt") {
			m_ptrBone = new DNN_torch;
		}
		else {
			m_ptrBone = new DNN_trt;
		}

		// input shape
		m_input_shape = NCHWshape(1, 3, 256, 256);

		// load model
		m_ptrBone->loadModel(
			model_filename,
			m_input_shape ,
			{ "output0", "output1", "output2", "output3", "output4" }
		);

		// output shape												C++	 (pt)		      C++	 (onnx)			   python
		m_output_heatmap = m_ptrBone->get_output_shape(0);		// (1,21,64,64)			(1,21,64,64)			//(1,21,64,64)
		m_output_so3 = m_ptrBone->get_output_shape(1);			//(1,48,-nan,-nan)		(1,10,-nan,-nan)		//(1,48)
		m_output_beta = m_ptrBone->get_output_shape(2);			//(1£¬10£¬-nan,-nan£©	(1, 1, 3, random)		//(1,10)
		m_output_joint_root = m_ptrBone->get_output_shape(3);   //(1,1,3,random)		(1£¬48£¬-nan,-nan£©		//(1,1,3)
		m_output_bone = m_ptrBone->get_output_shape(4);			//(1,1,1,random)		(1,1,1,random)			//(1,1,1)
	}
	~HandTailor() { if (m_ptrBone) delete m_ptrBone; }; 

protected:
	typedef Eigen::Vector4i NCHWshape;
	typedef Eigen::VectorXi TensorShape;

public:
	// \brief in this function,we can obtain heatmap, theta, beta, joint_root and bone from the input image
	// 
	// \param coords [in] the keypoint coords
	// \param so3_vec [in] theta param
	// \param beta [in] beta param
	// \param joint_root [in] unused
	// \param bone [in] unused
	// \param inputImage [out] the input image
	void get_param(
		std::vector<cv::Point2f>& coords,
		cv::Vec<float, 48>& so3_vec,
		cv::Vec<float, 10>& beta,
		cv::Vec<float, 3>& joint_root_vec,
		cv::Mat& bone,
		cv::Mat inputImage);

	// \brief obtain the scale according to the input param
	//
	// \param joints_3D [in] the 3D joints obtained from MANO model
	// \param joints_2D [in] the 2D joints obtained from heatmap ,and the realized in the function: hm_to_kp2d
	// 
	// \return the scale value, which will be used to form the projection matrix
	void obtain_scale(
		float& scale, 
		const Eigen::MatrixXf& joints_3D, 
		const std::vector<cv::Point2f>& joints_2D);

	// \brief obtain the bias between the real 3D joints and pred 3D joints
	//
	// \param theta [in] theta param
	// \param beta [in] beta param
	// \param joints3D [in] 3D joints,obtained from MANO model
	// \param joints2D [in] 2D joints,obtained from heatmap
	// \param t [in] the ninth channel of 2D joints
	// \param scale [in] scale value,calculated from the functions "obtain_scale()"
	//
	// \return the error between true value and predicted value
	float obtain_error(
		const cv::Vec<float, 48>& so3, 
		const cv::Vec<float, 10>& beta, 
		const MatrixXf& joints3D, 
		const std::vector<cv::Point2f>& joints2D, 
		const cv::Point2f& t, 
		float scale);

	// \brief convert the MANO results to aligen with the python
	//
	// \param Joints3D [out] the result after processing the MANO result ,and the number of the 3D joints convert to 21
	// \param Mesh_verts [out] the result after processing the MANO result of mesh verts
	// \param joints16 [in] MANO output ,represent the 3D joints
	// \param verts778 [in] MANO output ,represent the mesh verts
	void convertJointsandVerts(
		Eigen::MatrixXf& Joints3D ,
		Eigen::MatrixXf& Mesh_verts,
		const Eigen::MatrixXf& jonts16, 
		const Eigen::MatrixXf& verts778);

	// \brief draw the skeleton and 2d joints on the original
	//
	// \param inputImage [in] the original image
	// \param coords [in] the 2D joints' coords ,which are obtained in heatmap
	void draw(
		cv::Mat& inputImage, 
		const std::vector<cv::Point2f>& coords);

private:
	// \brief obtain the 2D joints from heatmap
	//
	// \param coords [out] the 2D joints' coords
	// \param heatmap [in] the net's output, and saved in a Mat type
	void obtain_2Dkeypoints(std::vector<cv::Point2f>& coords, const cv::Mat& heatmap);

	// \brief obtain joints loss
	//
	// \param joints_3D [in] 3D joints coords
	//
	// \return return the loss between the input 3D joints coords
	float obtain_loss(const Eigen::MatrixXf& joints_3D);

public:
	cv::Vec<float, 48> so3_init;
	cv::Vec<float, 10> beta_init;

private:
	DNNLoader*  m_ptrBone;
	NCHWshape	m_input_shape;
	NCHWshape	m_output_heatmap;
	NCHWshape	m_output_so3;
	NCHWshape	m_output_beta;
	NCHWshape	m_output_joint_root;
	NCHWshape	m_output_bone;
};
#endif // !__HAND_TAILOR_H
