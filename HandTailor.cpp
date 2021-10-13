#include "HandTailor.hpp"

void HandTailor::obtain_2Dkeypoints(std::vector<cv::Point2f>& coords, const cv::Mat& heatmap)
{
    float hm_sum = cv::sum(heatmap)[0];
    cv::Mat hm = heatmap / hm_sum;
    hm = hm.reshape(0, 1);
    float x_voc_buf;
    float y_voc_buf;
    int label = 0;

#pragma omp parallel for
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 64; j++) {
            x_voc_buf += hm.at<float>(label) * float(i);
            y_voc_buf += hm.at<float>(label) * float(j);
            label += 1;
        }
    }
    coords.push_back(cv::Point2f(y_voc_buf * 4, x_voc_buf * 4));  //the image's size is 256*256, and the heatmap's size is 64*64 ,so we should mul 4
}

void HandTailor::draw(cv::Mat& inputImage, const std::vector<cv::Point2f>& coords)
{
    int SKELETON[21][2] = { {0, 1},{1, 2},{2, 3}, {3, 4}, {0, 5}, {5, 6}, {6, 7},{7, 8},{0,9},
                            {9, 10}, {10, 11}, {11, 12}, {0, 13}, {13, 14}, {14, 15}, {15, 16},{0, 17},{17, 18},{18, 19},{19, 20} };

    cv::Scalar handColors[21] = {
            cv::Scalar(100, 100,100),
            cv::Scalar(219,10,219), cv::Scalar(219,10,219), cv::Scalar(219,10,219), cv::Scalar(219,10,219),
            cv::Scalar(255,0,0), cv::Scalar(255,0,0), cv::Scalar(255,0,0), cv::Scalar(255,0,0),
            cv::Scalar(0,255,0), cv::Scalar(0,255,0), cv::Scalar(0,255,0), cv::Scalar(0,255,0),
            cv::Scalar(0,255,255), cv::Scalar(0,255,255), cv::Scalar(0,255,255), cv::Scalar(0,255,255),
            cv::Scalar(0,0,255), cv::Scalar(0,0,255), cv::Scalar(0,0,255), cv::Scalar(0,0,255)
    };

     int thicknessLine = int(std::max((float)sqrt(inputImage.rows * inputImage.cols) / 150.f, 2.f));
     int radius = thicknessLine + 1;

     for (int i = 0; i < 21; i++) {
        std::map<int, cv::Point2f>      line;
        cv::Point2f                     point1;
        cv::Point2f                     point2;

        cv::circle(inputImage, coords[i], radius, handColors[i], -1);

        int line_a = SKELETON[i][0];
        int line_b = SKELETON[i][1];
        cv::line(inputImage, coords[line_a], coords[line_b], handColors[i], 2);
    }
}

void HandTailor::obtain_scale(float& scale, const Eigen::MatrixXf& joints_3D, const std::vector<cv::Point2f>& joints_2D)
{	
    cv::Point2f t = joints_2D[9];   //param t [in] the joint in ninth channel
    float s1 = 0;
    float s2 = 0;
    for (int i = 0; i < joints_2D.size(); i++) {
        s1 += joints_3D(0, i) * (joints_2D[i].x - t.x) + joints_3D(1, i) * (joints_2D[i].y - t.y);
        //cout << "µÚ" << to_string(i) << "Í¨µÀ:" << s1 << endl;
        s2 += joints_3D(0, i) * joints_3D(0, i) + joints_3D(1, i) * joints_3D(1, i);
    }
    scale = s1 / s2;
}

float HandTailor::obtain_loss(const Eigen::MatrixXf& joints_3D)
{
    int idx_a[5] = { 1,5,9,13,17 };
    int idx_b[5] = { 2,6,10,14,18 };
    int idx_c[5] = { 3,7,11,15,19 };
    int idx_d[5] = { 4,8,12,16,20 };

    MatrixXf p_a, p_b, p_c, p_d;
    std::vector<Vector3f> v_ab, v_bc, v_cd;
    for (int i = 0; i < 5; i++) {
        v_ab.push_back(joints_3D.col(idx_a[i]) - joints_3D.col(idx_b[i]));
        v_bc.push_back(joints_3D.col(idx_b[i]) - joints_3D.col(idx_c[i]));
        v_cd.push_back(joints_3D.col(idx_c[i]) - joints_3D.col(idx_d[i]));
    }

    float loss1, loss1_tmp;
    float loss2, loss2_tmp;
    //calculate loss1 and loss2
    for (int i = 0; i < 5; i++) {   
        loss1_tmp += abs((v_ab[i].cross(v_bc[i])).dot(v_cd[i]));       
        loss2_tmp += (v_ab[i].cross(v_bc[i])).dot(v_bc[i].cross(v_cd[i]));       
    }

    loss1 = loss1_tmp / 5.f;
    if (loss2_tmp < 0)
        loss2 = -loss2_tmp;
    else
        loss2 = 0;

    return 10000 * loss1 + 100000 * loss2;
}

float HandTailor::obtain_error(const cv::Vec<float, 48>& so3, const cv::Vec<float, 10>& beta, const MatrixXf& joints3D,
    const std::vector<cv::Point2f>& joints2D, const cv::Point2f& t, float scale)
{
    float           error;    
    float           reg_theta;
    float           reg_beta;
    float           errkp = 0;
    cv::Point2f     uv;
    MatrixXf        joints3D_tmp;

    float bone_pred = sqrt((joints3D.col(0) - joints3D.col(9)).dot(joints3D.col(0) - joints3D.col(9)));

    //calculate the mean bais of the theta and beta between the real value and init value
    reg_theta = ((so3 - so3_init).dot((so3 - so3_init)))/48.f;
    reg_beta = ((beta - beta_init).dot((beta - beta_init)))/10.f;

    joints3D_tmp = joints3D / bone_pred;
    for (int i = 0; i < joints3D.cols(); i++) {
        uv.x = (joints3D_tmp(0, i) * scale) + t.x;
        uv.y = (joints3D_tmp(1, i) * scale) + t.y;
        errkp += pow((uv.x - joints2D[i].x),2) + pow((uv.y - joints2D[i].y), 2);
    }
    float joint_loss = obtain_loss(joints3D);

    // final bais
    error = 0.01 * reg_theta + 0.01 * reg_beta + errkp / 42.f + 100 * joint_loss;

    return error;
}

void HandTailor::convertJointsandVerts(
    Eigen::MatrixXf& Joints3D,
    Eigen::MatrixXf& Mesh_verts,
    const Eigen::MatrixXf& jonts16,
    const Eigen::MatrixXf& verts778)
{
    std::vector<int> jtip_ind{ 745, 317, 444, 556, 673 };
    Eigen::MatrixXf jtip_pts = verts778(Eigen::all, jtip_ind);

    Eigen::MatrixXf jt_pts = Eigen::MatrixXf::Zero(3, jonts16.cols() + jtip_pts.cols());
    jt_pts << jonts16, jtip_pts;

    MatrixXf joint_tmp = MatrixXf::Zero(3, 21);;
    std::vector<int> color_order{ 0,13,14,15,16, 1,2,3,17, 4,5,6,18, 10,11,12,19, 7,8,9,20 }; // with tip_pts
    for (int i = 0; i < color_order.size(); i++) {
        joint_tmp.col(i) = jt_pts.col(color_order[i]);
    }

    Joints3D = joint_tmp.colwise() - joint_tmp.col(9);
    Mesh_verts = verts778.colwise() - joint_tmp.col(9);
}
void HandTailor::get_param(
    std::vector<cv::Point2f>& coords, 
    cv::Vec<float, 48>& so3_vec,
    cv::Vec<float, 10>& beta_vec,
    cv::Vec<float, 3>& joint_root_vec,
    cv::Mat& bone,
    cv::Mat inputImage)
{
    // allocate the memory
    float* ptr_input = new float[m_input_shape.prod()];
    float* ptr_output_heatmap = new float[m_output_heatmap.prod()];
    float* ptr_output_so3 = new float[m_output_so3.topRows(2).prod()];
    float* ptr_output_beta = new float[m_output_beta.topRows(2).prod()];
    float* ptr_output_joint_root = new float[m_output_joint_root.topRows(3).prod()];
    float* ptr_output_bone = new float[m_output_bone.topRows(3).prod()];

    std::vector<float*> ptr_output = { 
        ptr_output_heatmap, 
        ptr_output_so3,
        ptr_output_beta,
        ptr_output_joint_root,
        ptr_output_bone
    };

    // obtain the data for inference
    float ratio = m_ptrBone->img2Tensor(ptr_input, inputImage, m_input_shape);

    // inference 
    m_ptrBone->forward(ptr_output, ptr_input, m_input_shape);

    // get 2D joints from function get_2Djoints;
    for (int i = 0; i < 21; i++) {
        cv::Mat output_heatmap = cv::Mat(64, 64, CV_32FC1, ptr_output[0] + i * 64 * 64);
        obtain_2Dkeypoints(coords, output_heatmap);
    }

    // convert other model output's type
    for (int i = 0; i < 1; i++) {
        memcpy(&so3_vec, ptr_output[1], 192);   //pt ptr_output[1],onnx ptr_output[3]
        memcpy(&beta_vec, ptr_output[2],40);    //pt ptr_output[2],onnx ptr_output[1]
        memcpy(&joint_root_vec, ptr_output[3], 12);
        //joint_root = cv::Mat(3, 1, CV_32FC1, ptr_output[3] + i * 3 * 1); //pt ptr_output[3],onnx ptr_output[2]
        bone = cv::Mat(1, 1, CV_32FC1, ptr_output[4] + i * 1 * 1); //pt ptr_output[4],onnx ptr_output[4]

        so3_init = so3_vec;
        beta_init = beta_vec;
    }
}

