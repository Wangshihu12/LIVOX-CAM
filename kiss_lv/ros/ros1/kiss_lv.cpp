#include "kiss_lv.hpp"
#include <opencv2/xphoto/white_balance.hpp>
std::mutex mutex_image_callback;
LSDOptions opts;
cv::Mat prev_descriptors;
std::vector<cv::KeyPoint> prev_keypoints;
std::vector<Vec4f> prev_lines_ls;
constexpr double mid_pose_timestamp{0.5};
int sub_image_typed = 0; // 0: TBD 1: sub_raw, 2: sub_comp
Vector6dVector Get_FeatrueScan(const std::vector<Eigen::Vector3d>& laser_data,
                               const cv::Mat& intrisicMat,
                               const cv::Mat& extrinsicMat_RT,
                               const cv::Mat& feather_image);
Vector6dVector Get_map_cloud(const std::vector<Eigen::Vector3d>& laser_data,
                                           const cv::Mat& intrisicMat,
                                           const cv::Mat& extrinsicMat_RT,
                                           const cv::Mat& new_image);                    
Vector6dVector Trans_Vector6(const std::vector<Eigen::Vector3d>& laser_data);  
PointCloudXYZRGB::Ptr Eigen7dToPointCloud2(const std::vector<Vector7d> &points);     
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ConvertToXYZRGBPointCloud(const Vector6dVector& map_cloud);
cv::Mat ALTM_retinex(const cv::Mat& img);
cv::Mat get_image_keypoints(cv::Mat gray_image, int line_th, int line_len, int line_wide, int point_th, int radius_size);
double CalculatePointCloudVolume(Vector6dVector& frame, double voxel_size, int th_num, int &voxel_num);
std::vector<Eigen::Vector3d> PointCloud2ToEigen(const sensor_msgs::PointCloud2 &msg);
std::vector<double> Livox_time(const sensor_msgs::PointCloud2 &msg);
std::vector<Eigen::Vector3d> DeSkewScan(const std::vector<Eigen::Vector3d> &frame,
                                        const std::vector<double> &timestamps,
										const std::vector<Sophus::SE3d>& pose_deskew);
std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d> &frame);
std::vector<double> GetVelodyneTimestamps(const std::vector<Eigen::Vector3d> &points);
std::vector<double> GetTimestamps(const sensor_msgs::PointCloud2 &msg);
cv::Mat feather_image;
//-------------------------------------------------------------------------------------------------------------------------------------------
void KISS_LV::livox_handler(const livox_ros_driver2::CustomMsg::ConstPtr& livox_msg_in) {
	PointCloudXYZI::Ptr pcl_data(new PointCloudXYZI);
	auto time_end = livox_msg_in->points.back().offset_time;
    for (unsigned int i = 0; i < livox_msg_in->point_num; ++i) {
        PointTypeI pt;
        pt.x = livox_msg_in->points[i].x;
        pt.y = livox_msg_in->points[i].y;
        pt.z = livox_msg_in->points[i].z;
        double time = livox_msg_in->points[i].offset_time / (double)time_end;
        pt.intensity = time;
        pcl_data->push_back(pt);
    }
    sensor_msgs::PointCloud2 T_pointcloud;
    pcl::toROSMsg(*pcl_data, T_pointcloud);
    T_pointcloud.header = livox_msg_in->header;
    repub_point.publish(T_pointcloud);
}
void KISS_LV::ruby128_handler(const sensor_msgs::PointCloud::ConstPtr& cloud_msg) {
    sensor_msgs::PointCloud2 cloud2_msg;
    sensor_msgs::convertPointCloudToPointCloud2(*cloud_msg, cloud2_msg);
    repub_point.publish(cloud2_msg);
}

void KISS_LV::PointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
	double timestamp = msg->header.stamp.toSec();
	mtx_buffer.lock();
	ROS_DEBUG("get point cloud at time: %.6f", timestamp);
	if (timestamp < last_timestamp_lidar) 
	{
	    ROS_ERROR("lidar loop back, clear buffer");
	    lidar_buffer.clear();
	}
	last_timestamp_lidar = timestamp;
	lidar_buffer.push_back(msg);
	mtx_buffer.unlock();
	sig_buffer.notify_all();	
}

/**
 * @brief KISS_LV主循环函数 - 激光雷达视觉里程计(LiDAR Visual Odometry)核心处理函数
 * 
 * 该函数是KISS_LV系统的核心处理循环，负责：
 * 1. 同步激光雷达和相机数据
 * 2. 处理传感器数据并进行位姿估计
 * 3. 发布里程计信息和点云数据
 * 
 * 工作流程：
 * - 等待传感器数据同步完成
 * - 检查退出和重置标志
 * - 调用Register_Color_Frame进行帧注册和位姿估计
 * - 记录处理时间
 */
void KISS_LV::LVO(){
		// 主循环：当ROS节点运行时持续执行
		while (ros::ok()) {
			// 创建测量组对象，用于存储同步后的激光雷达和图像数据
			MeasureGroup meas;
			
			// 使用互斥锁保护缓冲区访问，确保线程安全
			std::unique_lock<std::mutex> lock(mtx_buffer);
			
			// 等待条件变量信号，直到数据同步完成或收到退出信号
			// Sync_packages函数负责同步激光雷达和相机数据
			// b_exit标志用于优雅退出程序
			sig_buffer.wait(lock, [this, &meas]() -> bool { return Sync_packages(meas) || b_exit; });
			
			// 释放互斥锁，允许其他线程访问缓冲区
			lock.unlock();
			
			// 检查退出标志，如果为true则退出主循环
			if (b_exit) 
			{
			    ROS_INFO("b_exit=true, exit");
			    break;
			}
			
			// 检查重置标志，通常在rosbag回放时使用
			// 当检测到时间戳回退时会设置此标志
			if (b_reset) 
			{
			    ROS_WARN("reset when rosbag play back");
			    b_reset = false;
			    continue;  // 跳过当前帧，重新开始处理
			}
			
			// 保存当前激光雷达消息的时间戳，用于后续的轨迹记录
			save_timestamp=meas.lidar_msg->header.stamp;
			
			// 记录处理开始时间，用于性能分析
			auto start = std::chrono::steady_clock::now();
			
			// 核心处理函数：注册彩色帧并进行位姿估计
			// 输入：同步后的激光雷达数据和图像数据
			// 功能：点云配准、位姿估计、地图更新
			KISS_LV::Register_Color_Frame(meas.lidar_msg, meas.image);
			
			// 记录处理结束时间
			auto end = std::chrono::steady_clock::now();
			
			// 计算处理耗时（毫秒）
			std::chrono::duration<double, std::milli> elapsed = end - start;
			
			// 输出处理时间信息，用于性能监控和调试
			std::cout << "Register_Color_Frame took " << elapsed.count() << " ms" << std::endl;
	}
}
bool KISS_LV::Sync_packages(MeasureGroup &measgroup) 
	{
		if (lidar_buffer.empty()) 
		{
			return false;
		}
		if (!lidar_pushed){
			//----------------------use_cam---------------------
			if (use_cam){
				if (image_buffer.empty()) 
				{
					return false;
				}
				if ((image_buffer.front()->header.stamp.toSec() > lidar_buffer.back()->header.stamp.toSec())) 
				{
					lidar_buffer.clear();
					ROS_ERROR("clear lidar buffer, only happen at the beginning===============");
					return false;
				}
			
				if (image_buffer.back()->header.stamp.toSec() < lidar_buffer.front()->header.stamp.toSec()) 
				{
					return false;
				}
			}
			lidar_pushed = true;
		}
		measgroup.lidar_msg = lidar_buffer.front();
		lidar_buffer.pop_front();
		double lidar_time = measgroup.lidar_msg->header.stamp.toSec();
		double min_time_diff = std::numeric_limits<double>::max();

		for (const auto &image : image_buffer) {
			double image_time = image->header.stamp.toSec();
			if (image_time <= lidar_time) {
				double time_diff = std::abs(lidar_time - image_time);
				if (time_diff < min_time_diff) {
				    min_time_diff = time_diff;
				    measgroup.image = image;
				}
				image_buffer.pop_front();
			}
		}
		lidar_pushed = false;
		return true;
}

void KISS_LV::ComImage_Callback(const sensor_msgs::CompressedImageConstPtr& msg)
{
    std::unique_lock<std::mutex> lock2(mutex_image_callback);
    if (sub_image_typed == 2) {
        return;
    }
    sub_image_typed = 1;
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    sensor_msgs::ImageConstPtr image_msg = cv_ptr->toImageMsg();
    if (msg->header.stamp.toSec() < last_timestamp_image) {
        ROS_ERROR("img loop back, clear buffer");
        mtx_buffer.lock();
        image_buffer.clear();
        mtx_buffer.unlock();
    }

    mtx_buffer.lock();
    image_buffer.push_back(image_msg);
    last_timestamp_image = msg->header.stamp.toSec();
    mtx_buffer.unlock();

    sig_buffer.notify_all();
}
void KISS_LV::Image_Callback(const sensor_msgs::ImageConstPtr& image_msg) {
    std::unique_lock<std::mutex> lock2(mutex_image_callback);
    if (sub_image_typed == 2) {
        return;
    }
    sub_image_typed = 1;
    if (image_msg->header.stamp.toSec() < last_timestamp_image) {
        ROS_ERROR("img loop back, clear buffer");
        mtx_buffer.lock();
        image_buffer.clear();
        mtx_buffer.unlock();
    }

    mtx_buffer.lock();
    image_buffer.push_back(image_msg);
    last_timestamp_image = image_msg->header.stamp.toSec();
    mtx_buffer.unlock();

    sig_buffer.notify_all();
}

/**
 * @brief 注册彩色帧函数 - KISS_LV系统的核心处理函数
 * 
 * 该函数负责激光雷达和相机数据的融合处理，包括：
 * 1. 多线程并行处理激光雷达和相机数据
 * 2. 点云去畸变和特征提取
 * 3. 位姿估计和地图更新
 * 4. 发布各种ROS消息和保存结果
 * 
 * @param lidar_msg 激光雷达点云消息
 * @param img_msg 相机图像消息
 */
void KISS_LV::Register_Color_Frame(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, const sensor_msgs::ImageConstPtr &img_msg) {
	// 记录处理开始时间，用于性能分析
	auto start = std::chrono::high_resolution_clock::now();
	
	// 初始化去畸变后的激光雷达扫描数据容器
	std::vector<Eigen::Vector3d> deskew_scan;
	deskew_scan.clear();
	
	// 初始化彩色点云和地图点云容器
	Vector6dVector color_cloud;    // 带颜色信息的点云 (x,y,z,r,g,b)
	Vector6dVector map_cloud;     // 用于地图构建的点云
	
	// 自适应体素化参数
	double adj_voxel_size, density;  // 调整后的体素大小和密度
	
	// 处理后的图像
	cv::Mat new_image;
	
	// ==================== 多传感器融合处理 ====================
	if (use_cam){
		// 使用多线程并行处理激光雷达和相机数据以提高效率
		// 激光雷达处理线程：去畸变、坐标变换等
		std::thread lidarThread([&]() { processLidarData(lidar_msg, deskew_scan); });
		
		// 相机处理线程：图像预处理、特征提取等
		std::thread cameraThread([&]() { processCameraData(img_msg, new_image, feather_image, intrisicMat_Resize); });
		
		// 等待两个线程完成
		lidarThread.join();
		cameraThread.join();
		
		// 基于特征图像生成彩色点云（用于里程计）
		color_cloud = Get_FeatrueScan(deskew_scan, intrisicMat_Resize, extrinsicMat_RT, feather_image);
		
		// 基于原始图像生成地图点云（用于可视化）
		map_cloud = Get_map_cloud(deskew_scan, intrisicMat, extrinsicMat_RT, new_image);
	}
	
	// ==================== 仅激光雷达模式 ====================
	if (!use_cam) {
		// 仅处理激光雷达数据
		processLidarData(lidar_msg, deskew_scan);
		
		// 将3D点云转换为6D点云（添加默认颜色）
		color_cloud = Trans_Vector6(deskew_scan);
		map_cloud = color_cloud;  // 地图点云与彩色点云相同
	}
	
	// ==================== 自适应空间模块 ====================
	// 根据点云密度自适应调整体素大小，提高配准精度
	const auto Input_scan = Adaptive_spatial_Module(color_cloud, adj_voxel_size, density);
	
	// ==================== 位姿估计 ====================
	// 使用KISS-LV算法进行帧间配准和位姿估计
	const auto keypoint = LVodometry_.RegisterFrame(Input_scan, adj_voxel_size, density);

	// 获取当前帧的位姿（SE3变换）
	const auto pose = LVodometry_.poses().back();
	
	// ==================== 性能统计 ====================
	auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // 将处理时间写入文件，用于性能分析
    std::ofstream foutC("/home/cxl/workspace/KISS_LV/src/kiss_lv/ros/results/cost_time.txt", std::ios::app);
	foutC.setf(std::ios::fixed, std::ios::floatfield);
	foutC.precision(3);
	foutC <<duration << std::endl;
	foutC.close();
	
	// ==================== 数据转换和准备 ====================
	// 获取局部地图用于可视化
	auto save_local_map_ = LVodometry_.LocalMap();
	
	// 将关键点转换为PCL点云格式
	scan_keypoint_enhance = Eigen7dToPointCloud2(keypoint);
	
	// 将局部地图转换为PCL点云格式
	save_map_points = Eigen7dToPointCloud2(save_local_map_);
	
	// 创建RGB点云用于可视化
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	rgb_cloud = ConvertToXYZRGBPointCloud(map_cloud);
	
	// 从位姿中提取平移和旋转信息
    const Eigen::Vector3d t_current = pose.translation();      // 平移向量
    const Eigen::Quaterniond q_current = pose.unit_quaternion(); // 四元数旋转

    // ==================== ROS消息发布 ====================
    
    // 发布TF变换消息（用于坐标变换）
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.stamp = ros::Time::now();
    transform_msg.header.frame_id = odom_frame_;      // 父坐标系
    transform_msg.child_frame_id = child_frame_;     // 子坐标系
    transform_msg.transform.rotation.x = q_current.x();
    transform_msg.transform.rotation.y = q_current.y();
    transform_msg.transform.rotation.z = q_current.z();
    transform_msg.transform.rotation.w = q_current.w();
    transform_msg.transform.translation.x = t_current.x();
    transform_msg.transform.translation.y = t_current.y();
    transform_msg.transform.translation.z = t_current.z();
    tf_broadcaster_.sendTransform(transform_msg);

    // 发布里程计消息
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = save_timestamp;  // 使用激光雷达时间戳
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = child_frame_;
    odom_msg.pose.pose.orientation.x = q_current.x();
    odom_msg.pose.pose.orientation.y = q_current.y();
    odom_msg.pose.pose.orientation.z = q_current.z();
    odom_msg.pose.pose.orientation.w = q_current.w();
    odom_msg.pose.pose.position.x = t_current.x();
    odom_msg.pose.pose.position.y = t_current.y();
    odom_msg.pose.pose.position.z = t_current.z();
    odom_publisher_.publish(odom_msg);
	
    // ==================== 轨迹保存 ====================
    // 保存TUM格式的轨迹文件（用于评估）
    if (Save_path){
    	std::ofstream foutC(string(string(ROOT_DIR) + "/kiss_lv.txt"), std::ios::app);
		foutC.setf(std::ios::fixed, std::ios::floatfield);
		foutC.precision(5);
		foutC << save_timestamp << " "
		      << odom_msg.pose.pose.position.x << " "
		      << odom_msg.pose.pose.position.y << " "
		      << odom_msg.pose.pose.position.z << " "
		      << odom_msg.pose.pose.orientation.x << " "
		      << odom_msg.pose.pose.orientation.y << " "
		      << odom_msg.pose.pose.orientation.z << " "
		      << odom_msg.pose.pose.orientation.w << std::endl;
		foutC.close();
    }
    
    // ==================== 轨迹可视化 ====================
    // 发布轨迹消息（每两帧发布一次以降低频率）
    scan_num++;
    if (scan_num%2==0){
		geometry_msgs::PoseStamped pose_msg;
		pose_msg.pose = odom_msg.pose.pose;
		pose_msg.header = odom_msg.header;
		path_msg_.poses.push_back(pose_msg);
		traj_publisher_.publish(path_msg_);
	}
	
	// ==================== 点云数据发布 ====================
	// 发布RGB点云（当前帧的彩色点云）
	sensor_msgs::PointCloud2 rgb_pointscan;
	pcl::toROSMsg(*rgb_cloud, rgb_pointscan);
	rgb_pointscan.header.stamp = ros::Time::now();
	rgb_pointscan.header.frame_id = child_frame_;
	frame_publisher_.publish(rgb_pointscan);
	
	// 发布关键点点云（用于调试和可视化）
	sensor_msgs::PointCloud2 key_pointcloud;
	pcl::toROSMsg(*scan_keypoint_enhance, key_pointcloud);
	key_pointcloud.header.stamp = ros::Time::now();
	key_pointcloud.header.frame_id = child_frame_;
	kpoints_publisher_.publish(key_pointcloud);
	
	// 发布局部地图点云
    sensor_msgs::PointCloud2 map_msg;
	pcl::toROSMsg(*save_map_points, map_msg);	
	map_msg.header.stamp = ros::Time::now();
	map_msg.header.frame_id = odom_frame_;
	local_map_publisher_.publish(map_msg);
	
	// ==================== 图像数据发布 ====================
	// 发布特征图像（用于调试和可视化）
	sensor_msgs::ImagePtr featherimage_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", feather_image).toImageMsg();
	featherimage_pub.publish(featherimage_msg);
	
	// ==================== 点云地图保存 ====================
	// 保存完整的点云地图到PLY文件
	if (rgb_cloud->size() > 0 && pcd_save_en)
	{
		pcd_index++;
		if (pcd_index >= save_frame_num_beg){
			// 将当前帧点云变换到全局坐标系
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			for (const auto& point : rgb_cloud->points)
			{
				// 使用当前位姿变换点云到全局坐标系
				Eigen::Vector3d transformed_point = (pose * Eigen::Vector3d(point.x, point.y, point.z)).cast<double>();
				pcl::PointXYZRGB transformed_pcl_point;
				transformed_pcl_point.x = transformed_point.x();
				transformed_pcl_point.y = transformed_point.y();
				transformed_pcl_point.z = transformed_point.z();
				transformed_pcl_point.r = point.r;
				transformed_pcl_point.g = point.g;
				transformed_pcl_point.b = point.b;
				transformed_rgb_cloud->push_back(transformed_pcl_point);
			}
			// 将变换后的点云添加到完整地图中
			*complete_map += *transformed_rgb_cloud;
			
			// 生成保存路径
			string all_points_dir(string(string(ROOT_DIR) + "/scans_") + to_string(save_frame_num_end) + string(".ply"));
			pcl::PLYWriter ply_writer;
			
			// 当达到指定帧数时保存完整地图
			if (pcd_index == save_frame_num_end)
			{
			    cout << "current scan saved to /PLY/:" << all_points_dir << endl;
			    ply_writer.write(all_points_dir, *complete_map);
			}
	    }
	}
	
	// ==================== 清理和重置 ====================
	// 重置参数，为下一帧处理做准备
	KISS_LV::resetParameters();
}

/**
 * @brief 激光雷达数据处理函数 - 负责激光雷达点云的预处理和去畸变
 * 
 * 该函数的主要功能：
 * 1. 将ROS点云消息转换为Eigen格式
 * 2. 根据配置决定是否进行去畸变处理
 * 3. 针对不同激光雷达类型进行相应的处理
 * 
 * 去畸变(Deskew)的作用：
 * - 激光雷达在扫描过程中，传感器本身在运动
 * - 不同时刻采集的点具有不同的传感器位姿
 * - 去畸变将同一帧内的所有点投影到同一时刻的坐标系
 * - 提高点云配准的精度和稳定性
 * 
 * @param lidar_msg 输入的激光雷达点云消息
 * @param deskew_scan 输出的去畸变后的点云数据（引用传递）
 */
void KISS_LV::processLidarData(const sensor_msgs::PointCloud2::ConstPtr& lidar_msg, std::vector<Eigen::Vector3d> &deskew_scan) {
	// 将ROS PointCloud2消息转换为Eigen::Vector3d格式的点云
	// 这一步提取了点云的几何信息（x, y, z坐标）
	const auto points = PointCloud2ToEigen(*lidar_msg);
	
	// ==================== 去畸变处理 ====================
	// 检查配置文件中是否启用了去畸变功能
	if (config_.deskew){
		// 获取历史位姿序列，用于运动补偿
		// poses()返回所有已估计的位姿，用于插值计算中间时刻的位姿
		const auto pose_deskew = LVodometry_.poses();
		
		// 针对LIVOX激光雷达的特殊处理
		// LIVOX激光雷达具有非重复扫描模式，需要特殊的时间戳处理
		if (config_.type == "LIVOX"){
			// 从LIVOX点云中提取时间戳信息
			// LIVOX激光雷达在intensity字段中存储相对时间戳
			const auto timestamps = Livox_time(*lidar_msg);
			
			// 执行去畸变处理
			// DeSkewScan函数使用历史位姿和时间戳进行运动补偿
			// 将所有点投影到扫描开始时刻的坐标系
			deskew_scan = DeSkewScan(points, timestamps, pose_deskew);
		}
		// 注意：这里只处理了LIVOX类型，其他激光雷达类型（如Velodyne）可能需要不同的处理方式
	}
	
	// ==================== 无去畸变模式 ====================
	// 如果配置中禁用了去畸变，直接使用原始点云数据
	if (!config_.deskew){
		// 直接将原始点云赋值给输出，不进行任何运动补偿
		deskew_scan = points;
	}
	
	// 输出说明：
	// - deskew_scan现在包含了处理后的点云数据
	// - 如果启用了去畸变，点云已经过运动补偿
	// - 如果未启用去畸变，点云保持原始状态
	// - 后续的配准算法将使用这些处理后的点云进行位姿估计
}

/**
 * @brief 相机数据处理函数 - 负责相机图像的预处理、增强和特征提取
 * 
 * 该函数的主要功能：
 * 1. 图像去畸变和校正
 * 2. 图像增强和预处理
 * 3. 白平衡调整
 * 4. 特征提取（线条和关键点）
 * 
 * 处理流程：
 * ROS图像消息 → OpenCV格式 → 去畸变 → 图像增强 → 缩放 → 白平衡 → 灰度化 → 特征提取
 * 
 * @param msg 输入的ROS图像消息
 * @param new_image 输出的增强后图像（引用传递）
 * @param feather_image 输出的特征图像，包含线条和关键点（引用传递）
 * @param intrisicMat_Resize 缩放后的内参矩阵（用于后续处理）
 */
void KISS_LV::processCameraData(const sensor_msgs::ImageConstPtr &msg,
                                 cv::Mat &new_image,
                                 cv::Mat &feather_image,
                                 cv::Mat intrisicMat_Resize)
{
    try {
        // ==================== 静态变量初始化 ====================
        // 去畸变映射表，用于快速去畸变处理
        // 使用static确保只初始化一次，提高效率
        static cv::Mat map1, map2;                    // 去畸变映射表
        static bool map_initialized = false;         // 映射表初始化标志
        
        // 白平衡处理器，用于自动调整图像色彩平衡
        static cv::Ptr<cv::xphoto::SimpleWB> wb = cv::xphoto::createSimpleWB();
        wb->setInputMin(0.0f);    // 设置输入最小值
        wb->setInputMax(255.0f);  // 设置输入最大值

        // ==================== 图像格式转换 ====================
        // 将ROS图像消息转换为OpenCV Mat格式
        // cv_bridge是ROS和OpenCV之间的桥梁
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        const cv::Mat& raw_img = cv_ptr->image;  // 获取原始图像

        // ==================== 去畸变映射表初始化 ====================
        // 只在第一次调用时初始化映射表，避免重复计算
        if (!map_initialized) {
            // 初始化去畸变和校正映射表
            // un_intrisicMat: 去畸变后的内参矩阵
            // distCoeffs: 畸变系数
            // raw_img.size(): 图像尺寸
            // CV_16SC2: 映射表数据类型
            cv::initUndistortRectifyMap(un_intrisicMat, distCoeffs, cv::Mat(),
                                        un_intrisicMat, raw_img.size(), CV_16SC2, map1, map2);
            map_initialized = true;  // 标记已初始化
        }

        // ==================== 图像去畸变 ====================
        // 使用预计算的映射表进行快速去畸变
        cv::Mat undistorted;
        cv::remap(raw_img, undistorted, map1, map2, cv::INTER_LINEAR);
        
        // ==================== 图像增强 ====================
        // 使用ALTM_retinex算法进行图像增强
        // Retinex算法可以改善光照不均和对比度问题
        new_image = ALTM_retinex(undistorted);

        // ==================== 图像缩放 ====================
        // 将图像缩放到较小尺寸以提高处理速度
        cv::Mat small_img;
        cv::resize(new_image, small_img, cv::Size(W / resize, H / resize));
        
        // ==================== 白平衡调整 ====================
        // 自动调整图像的白平衡，改善色彩表现
        wb->balanceWhite(small_img, small_img);

        // ==================== 灰度化和滤波 ====================
        // 转换为灰度图像用于特征提取
        cv::Mat gray;
        cv::cvtColor(small_img, gray, cv::COLOR_BGR2GRAY);
        
        // 中值滤波去除噪声，保持边缘信息
        cv::medianBlur(gray, gray, 3);
        
        // ==================== 特征提取 ====================
        // 从灰度图像中提取线条特征和关键点特征
        // 参数说明：
        // - line_th: 线条检测阈值
        // - line_len: 最小线条长度
        // - line_wide: 线条绘制宽度
        // - point_th: 关键点数量阈值
        // - radius_size: 关键点绘制半径
        feather_image = get_image_keypoints(gray, line_th, line_len, line_wide, point_th, radius_size);
    }
    catch (cv_bridge::Exception &e) {
        // ==================== 异常处理 ====================
        // 如果图像格式转换失败，输出错误信息
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
    
    // 输出说明：
    // - new_image: 经过去畸变、增强、缩放、白平衡处理的彩色图像
    // - feather_image: 包含线条和关键点特征的可视化图像
    // - 这些图像将用于后续的点云着色和特征匹配
}


void KISS_LV::resetParameters(){ 
	  good_points->clear();
	  scan_keypoint_enhance->clear();
	  color_cloud->clear();
	  map_cloud->clear();
	  save_map_points->clear();
}

std::vector<Eigen::Vector3d> PointCloud2ToEigen(const sensor_msgs::PointCloud2 &msg) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(msg.height * msg.width);
    sensor_msgs::PointCloud2ConstIterator<float> msg_x(msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> msg_y(msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> msg_z(msg, "z");
    for (size_t i = 0; i < msg.height * msg.width; ++i, ++msg_x, ++msg_y, ++msg_z) {
        points.emplace_back(*msg_x, *msg_y, *msg_z);
    }
    return points;
}
std::vector<double> Livox_time(const sensor_msgs::PointCloud2 &msg) {
    std::vector<double> timestamps;
    timestamps.reserve(msg.height * msg.width);


    sensor_msgs::PointCloud2ConstIterator<float> intensity_it(msg, "intensity");

    for (size_t i = 0; i < msg.height * msg.width; ++i, ++intensity_it) {

        timestamps.push_back(static_cast<double>(*intensity_it));
    }

    return timestamps;
}
std::vector<Eigen::Vector3d> DeSkewScan(const std::vector<Eigen::Vector3d> &frame,
                                        const std::vector<double> &timestamps,
										const std::vector<Sophus::SE3d>& pose_deskew) {
    
    const size_t N = pose_deskew.size();
    if (N <= 2) return frame;
    const auto& start_pose = pose_deskew[N - 2];
    const auto& finish_pose = pose_deskew[N - 1];
    const auto delta_pose = (start_pose.inverse() * finish_pose).log();
    if (!delta_pose.allFinite()) {
        std::cerr << "[DeSkewScan] Invalid delta_pose (NaN or Inf detected)" << std::endl;
        return frame;
    }
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        const auto motion = Sophus::SE3d::exp((timestamps[i] - mid_pose_timestamp) * delta_pose);
        corrected_frame[i] = motion * frame[i];
    });
    return corrected_frame;
}
auto GetTimestampField(const sensor_msgs::PointCloud2 &msg) {
    PointField timestamp_field;
    for (const auto &field : msg.fields) {
        if ((field.name == "t" || field.name == "timestamp" || field.name == "time")) {
            timestamp_field = field;
        }
    }
    if (!timestamp_field.count) {
        throw std::runtime_error("Field 't', 'timestamp', or 'time'  does not exist");
    }
    return timestamp_field;
}
auto NormalizeTimestamps(const std::vector<double> &timestamps) {
    const double max_timestamp = *std::max_element(timestamps.cbegin(), timestamps.cend());
    // check if already normalized
    if (max_timestamp < 1.0) return timestamps;
    std::vector<double> timestamps_normalized(timestamps.size());
    std::transform(timestamps.cbegin(), timestamps.cend(), timestamps_normalized.begin(),
                   [&](const auto &timestamp) { return timestamp / max_timestamp; });
    return timestamps_normalized;
}
auto ExtractTimestampsFromMsg(const sensor_msgs::PointCloud2 &msg, const PointField &field) {
    // Extract timestamps from cloud_msg
    const size_t n_points = msg.height * msg.width;
    std::vector<double> timestamps;
    timestamps.reserve(n_points);

    // Option 1: Timestamps are unsigned integers -> epoch time.
    if (field.name == "t" || field.name == "timestamp") {
        sensor_msgs::PointCloud2ConstIterator<uint32_t> msg_t(msg, field.name);
        for (size_t i = 0; i < n_points; ++i, ++msg_t) {
            timestamps.emplace_back(static_cast<double>(*msg_t));
        }
        // Covert to normalized time, between 0.0 and 1.0
        return NormalizeTimestamps(timestamps);
    }

    // Option 2: Timestamps are floating point values between 0.0 and 1.0
    // field.name == "timestamp"
    sensor_msgs::PointCloud2ConstIterator<double> msg_t(msg, field.name);
    for (size_t i = 0; i < n_points; ++i, ++msg_t) {
        timestamps.emplace_back(*msg_t);
    }
    return timestamps;
}
std::vector<double> GetTimestamps(const sensor_msgs::PointCloud2 &msg) {
    auto timestamp_field = GetTimestampField(msg);
    std::vector<double> timestamps = ExtractTimestampsFromMsg(msg, timestamp_field);

    return timestamps;
}


std::vector<double> GetVelodyneTimestamps(const std::vector<Eigen::Vector3d> &points) {
    std::vector<double> timestamps;
    timestamps.reserve(points.size());
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        const double yaw = -std::atan2(point.y(), point.x());
        timestamps.emplace_back(0.5 * (yaw / M_PI + 1.0));
    });
    return timestamps;
}
std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d> &frame) {
    constexpr double VERTICAL_ANGLE_OFFSET = (0.205 * M_PI) / 180.0;
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        const auto &pt = frame[i];
        const Eigen::Vector3d rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.));
        corrected_frame[i] =
            Eigen::AngleAxisd(VERTICAL_ANGLE_OFFSET, rotationVector.normalized()) * pt;
    });
    return corrected_frame;
}
/**
 * @brief 特征扫描函数 - 将激光雷达点云投影到特征图像上并根据特征信息着色
 * 
 * 该函数的主要功能：
 * 1. 将3D激光雷达点云投影到2D特征图像上
 * 2. 根据特征图像中的特征信息为点云着色
 * 3. 生成带颜色信息的6D点云(x,y,z,r,g,b)
 * 
 * 投影过程：
 * 激光雷达坐标系 → 相机坐标系 → 图像坐标系
 * 
 * 着色策略：
 * - 默认颜色：绿色(55,100,55)
 * - 特征颜色：根据特征图像中的线条和关键点信息着色
 * - 特征检测：通过特定的BGR颜色组合识别特征区域
 * 
 * @param laser_data 输入的激光雷达3D点云数据
 * @param intrisicMat 相机内参矩阵(3x3)
 * @param extrinsicMat_RT 激光雷达到相机的外参变换矩阵(4x4)
 * @param feather_image 特征图像，包含线条和关键点的可视化信息
 * @return Vector6dVector 带颜色信息的6D点云(x,y,z,r,g,b)
 */
Vector6dVector Get_FeatrueScan(const std::vector<Eigen::Vector3d>& laser_data,
                               const cv::Mat& intrisicMat,
                               const cv::Mat& extrinsicMat_RT,
                               const cv::Mat& feather_image) {
    // ==================== 图像尺寸获取 ====================
    int H = feather_image.rows;  // 特征图像高度
    int W = feather_image.cols;  // 特征图像宽度
    
    // 存储最终输出的6D点云(x,y,z,r,g,b)
    Vector6dVector pl_points;

    // ==================== 矩阵格式转换 ====================
    // 将OpenCV Mat格式转换为Eigen Matrix格式，便于数学运算
    Eigen::Matrix<double, 3, 3> intrinsic;   // 相机内参矩阵
    Eigen::Matrix<double, 4, 4> extrinsic;   // 激光雷达到相机的变换矩阵
    cv::cv2eigen(intrisicMat, intrinsic);    // OpenCV → Eigen转换
    cv::cv2eigen(extrinsicMat_RT, extrinsic);

    // ==================== 并行点云处理 ====================
    // 使用OpenMP并行处理点云，提高处理速度
    #pragma omp parallel
    {
        // 每个线程维护自己的局部点云容器，避免数据竞争
        Vector6dVector local_points;

        // 并行遍历所有激光雷达点
        #pragma omp for nowait
        for (int i = 0; i < laser_data.size(); ++i) {
            const auto& pt = laser_data[i];  // 获取当前3D点

            // ==================== 距离过滤 ====================
            // 过滤掉距离过近的点（可能是噪声或无效数据）
            if (pt[0] <= 0)
                continue;

            // ==================== 坐标变换 ====================
            // 将激光雷达坐标系下的点转换为齐次坐标
            Eigen::Vector4d pointLidar(pt[0], pt[1], pt[2], 1.0);
            
            // 使用外参矩阵将点从激光雷达坐标系变换到相机坐标系
            Eigen::Vector4d tempPoint = extrinsic * pointLidar;

            // ==================== 深度过滤 ====================
            // 过滤掉相机坐标系下深度为负的点（在相机后方）
            if (tempPoint[2] <= 0) continue;

            // ==================== 投影到图像平面 ====================
            // 使用相机内参将3D点投影到2D图像平面
            // 透视投影公式：u = fx * X/Z + cx, v = fy * Y/Z + cy
            Eigen::Vector3d imgPoint = intrinsic * tempPoint.head<3>() / tempPoint[2];

            // 将浮点坐标转换为整数像素坐标
            int u = static_cast<int>(imgPoint[0]);  // 图像x坐标
            int v = static_cast<int>(imgPoint[1]);  // 图像y坐标
            
            // ==================== 边界检查和着色 ====================
            // 检查投影点是否在图像边界内
            if (u >= 0 && u < W && v >= 0 && v < H) {
                // 设置默认颜色：绿色(55,100,55)
                Eigen::Vector3d color(55, 100, 55);

                // ==================== 特征检测和着色 ====================
                // 获取特征图像中对应像素的BGR值
                const uchar* pixel = feather_image.ptr<uchar>(v) + 3 * u;
                uchar b = pixel[0];  // 蓝色分量
                uchar g = pixel[1];  // 绿色分量
                uchar r = pixel[2];   // 红色分量

                // 检测特征区域：通过特定的BGR颜色组合识别线条和关键点
                // 条件1：(b==255 && r==55) - 检测到某种特征
                // 条件2：(b==55 && r==255) - 检测到另一种特征
                if ((b == 255 && r == 55) || (b == 55 && r == 255)) {
                    // 根据特征类型设置特殊颜色
                    color[0] = static_cast<double>(r);  // 红色分量
                    color[1] = 55.0;                     // 绿色分量固定为55
                    color[2] = static_cast<double>(b);  // 蓝色分量
                }

                // ==================== 构建6D点云 ====================
                // 将3D坐标和颜色信息组合成6D点
                Vector6d point;
                point << pt, color;  // 前3维是坐标(x,y,z)，后3维是颜色(r,g,b)
                
                // 将处理后的点添加到局部容器
                local_points.push_back(point);
            }
        }
        
        // ==================== 线程同步 ====================
        // 使用critical section将各线程的局部结果合并到全局容器
        #pragma omp critical
        pl_points.insert(pl_points.end(), local_points.begin(), local_points.end());
    }

    // ==================== 返回结果 ====================
    // 返回包含所有有效投影点的6D彩色点云
    // 该点云将用于：
    // 1. 激光雷达视觉里程计的配准
    // 2. 特征增强的点云配准
    // 3. 多传感器融合的位姿估计
    return pl_points;
}
/**
 * @brief 地图点云生成函数 - 将激光雷达点云投影到原始图像上并根据真实颜色着色
 * 
 * 该函数的主要功能：
 * 1. 将3D激光雷达点云投影到2D原始图像上
 * 2. 根据图像的真实RGB颜色为点云着色
 * 3. 生成用于地图构建的6D彩色点云(x,y,z,r,g,b)
 * 
 * 与Get_FeatrueScan的区别：
 * - Get_FeatrueScan：使用特征图像，根据特征信息着色（用于里程计）
 * - Get_map_cloud：使用原始图像，根据真实颜色着色（用于地图构建）
 * 
 * 投影过程：
 * 激光雷达坐标系 → 相机坐标系 → 图像坐标系
 * 
 * 着色策略：
 * - 直接使用原始图像的RGB值
 * - 保持图像的真实色彩信息
 * - 用于生成高质量的可视化地图
 * 
 * @param laser_data 输入的激光雷达3D点云数据
 * @param intrisicMat 相机内参矩阵(3x3)
 * @param extrinsicMat_RT 激光雷达到相机的外参变换矩阵(4x4)
 * @param new_image 原始彩色图像，包含真实的RGB颜色信息
 * @return Vector6dVector 带真实颜色信息的6D点云(x,y,z,r,g,b)
 */
Vector6dVector Get_map_cloud(const std::vector<Eigen::Vector3d>& laser_data,
                             const cv::Mat& intrisicMat,
                             const cv::Mat& extrinsicMat_RT,
                             const cv::Mat& new_image) {
    // ==================== 图像尺寸获取 ====================
    int H = new_image.rows;  // 原始图像高度
    int W = new_image.cols;  // 原始图像宽度
    
    // 存储最终输出的6D彩色点云(x,y,z,r,g,b)
    Vector6dVector rgb_points;

    // ==================== 矩阵格式转换 ====================
    // 将OpenCV Mat格式转换为Eigen Matrix格式，便于数学运算
    Eigen::Matrix3d intrinsic;   // 相机内参矩阵(3x3)
    Eigen::Matrix4d extrinsic;   // 激光雷达到相机的变换矩阵(4x4)
    cv::cv2eigen(intrisicMat, intrinsic);    // OpenCV → Eigen转换
    cv::cv2eigen(extrinsicMat_RT, extrinsic);

    // ==================== 并行点云处理 ====================
    // 使用OpenMP并行处理点云，提高处理速度
    #pragma omp parallel
    {
        // 每个线程维护自己的局部点云容器，避免数据竞争
        Vector6dVector local_points;

        // 并行遍历所有激光雷达点
        #pragma omp for nowait
        for (int i = 0; i < laser_data.size(); ++i) {
            const auto& pt = laser_data[i];  // 获取当前3D点
            
            // ==================== 坐标变换 ====================
            // 将激光雷达坐标系下的点转换为齐次坐标
            Eigen::Vector4d pt_lidar(pt.x(), pt.y(), pt.z(), 1.0);
            
            // 使用外参矩阵将点从激光雷达坐标系变换到相机坐标系
            Eigen::Vector4d pt_cam = extrinsic * pt_lidar;
            
            // ==================== 深度过滤 ====================
            // 过滤掉相机坐标系下深度为负的点（在相机后方）
            if (pt_cam[2] <= 0) continue;

            // ==================== 投影到图像平面 ====================
            // 使用相机内参将3D点投影到2D图像平面
            // 透视投影公式：u = fx * X/Z + cx, v = fy * Y/Z + cy
            Eigen::Vector3d pt_img = intrinsic * pt_cam.head<3>() / pt_cam[2];

            // 将浮点坐标转换为整数像素坐标
            int u = static_cast<int>(pt_img[0]);  // 图像x坐标
            int v = static_cast<int>(pt_img[1]);  // 图像y坐标

            // ==================== 边界检查和着色 ====================
            // 检查投影点是否在图像边界内
            if (u >= 0 && u < W && v >= 0 && v < H) {
                // ==================== 真实颜色提取 ====================
                // 获取原始图像中对应像素的真实RGB值
                const uchar* pixel = new_image.ptr<uchar>(v) + 3 * u;
                uchar b = pixel[0];  // 蓝色分量
                uchar g = pixel[1];  // 绿色分量
                uchar r = pixel[2];   // 红色分量

                // ==================== 构建6D彩色点云 ====================
                // 将3D坐标和真实RGB颜色信息组合成6D点
                Vector6d rgb_point;
                rgb_point << pt.x(), pt.y(), pt.z(),           // 前3维：3D坐标(x,y,z)
                             static_cast<double>(r),            // 第4维：红色分量
                             static_cast<double>(g),            // 第5维：绿色分量
                             static_cast<double>(b);           // 第6维：蓝色分量
                
                // 将处理后的彩色点添加到局部容器
                local_points.push_back(rgb_point);
            }
        }
        
        // ==================== 线程同步 ====================
        // 使用critical section将各线程的局部结果合并到全局容器
        #pragma omp critical
        rgb_points.insert(rgb_points.end(), local_points.begin(), local_points.end());
    }
    
    // ==================== 返回结果 ====================
    // 返回包含所有有效投影点的6D彩色点云
    // 该点云将用于：
    // 1. 高质量地图构建和可视化
    // 2. 真实色彩的点云地图展示
    // 3. 后续的地图保存和导出
    return rgb_points;
}
Vector6dVector Trans_Vector6(const std::vector<Eigen::Vector3d>& laser_data){
	std::vector<Vector6d> laser_data_extended;
	laser_data_extended.reserve(laser_data.size());
	for (const auto& point : laser_data) {
		Vector6d extended_point;
		extended_point << point, 55, 150, 55;
		laser_data_extended.push_back(extended_point);
	}
	return laser_data_extended;
}

Vector6dVector KISS_LV::Adaptive_spatial_Module(const Vector6dVector& color_cloud, double &adj_voxel_size, double &density) {
    Vector6dVector filtered_vector_array;
    int point_num = 0;
    density = 0.0;
    adj_voxel_size = 0.0;
    
    for (const auto& vector : color_cloud) {
        double distance = std::sqrt(vector(0) * vector(0) + vector(1) * vector(1) + vector(2) * vector(2));
        if (distance > config_.min_range && distance < config_.max_range) {
            filtered_vector_array.push_back(vector);
            point_num++;
        }
    }
    int voxel_num;
    double exp_voxel_num = point_num/exp_key_num;
    double init_volume = CalculatePointCloudVolume(filtered_vector_array, config_.voxel_size, 3, voxel_num);
    density = ceil(point_num / voxel_num);
    double vc = pow(exp_voxel_num*config_.voxel_size*config_.voxel_size*config_.voxel_size/density, 1.0/3.0);
    if (!filtered_vector_array.empty()) {
        adj_voxel_size = vc;
    }
    else {
        adj_voxel_size = config_.voxel_size;
        density = 20;
    }
    std::cout<<"+++++++++++++++"<<std::endl;
    cout<<"point_num--"<<point_num<<endl;
    cout<<"adj_voxel_size--"<<adj_voxel_size<<endl;
    cout<<"density--"<<density<<endl;
    return filtered_vector_array;
}

struct VoxelInfo {
    std::set<int> pointIndices;
    bool isSparse = false;
};
struct Vector3iComparator {
    bool operator()(const Eigen::Vector3i& lhs, const Eigen::Vector3i& rhs) const {
        for (int i = 0; i < 3; ++i) {
            if (lhs[i] < rhs[i]) return true;
            if (lhs[i] > rhs[i]) return false;
        }
        return false;
    }
};
double CalculatePointCloudVolume(Vector6dVector& frame, double voxel_size, int min_neighbors, int &voxel_num) {
    voxel_num = 0;
    std::map<Eigen::Vector3i, VoxelInfo, Vector3iComparator> voxel_info_map;
    for (int i = 0; i < frame.size(); ++i) {
        Eigen::Vector3i voxel_indices = (frame[i].head<3>() / voxel_size).cast<int>();
        voxel_info_map[voxel_indices].pointIndices.insert(i);
    }
	for (auto& voxel : voxel_info_map) {
		int point_count = voxel.second.pointIndices.size();
		voxel.second.isSparse = (point_count <= min_neighbors);
	}
    double total_voxel_volume = voxel_size * voxel_size * voxel_size;
    double covered_volume = 0.0;
    Vector6dVector non_sparse_point_cloud;
    for (const auto& voxel : voxel_info_map) {
        if (!voxel.second.isSparse) {
        	voxel_num++;
            covered_volume += total_voxel_volume;
            for (int index : voxel.second.pointIndices) {
                non_sparse_point_cloud.push_back(frame[index]);
            }
        }
    }
    frame = non_sparse_point_cloud;
    return covered_volume;
}

PointCloudXYZRGB::Ptr Eigen7dToPointCloud2(const std::vector<Vector7d> &points) {
    PointCloudXYZRGB::Ptr output_cloud(new PointCloudXYZRGB()); 
    for (const auto& point : points)
    {
        ColorPointType pcl_point;
        pcl_point.x = point[0];
        pcl_point.y = point[1];
        pcl_point.z = point[2];
        pcl_point.r = point[3];
        pcl_point.g = point[4];
        pcl_point.b = point[5];
        output_cloud->push_back(pcl_point);
    }
    return output_cloud;
}
cv::Mat ALTM_retinex(const cv::Mat& img)
{
    cv::Mat ycrcb;
    cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels(3);
    cv::split(ycrcb, channels);

    double tile_size = std::max(img.cols * 32.0 / 1280.0, 4.0);
    cv::Size tile_grid_size(static_cast<int>(tile_size), static_cast<int>(tile_size));

    static thread_local cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(1.0, tile_grid_size);

    clahe->apply(channels[0], channels[0]);

    cv::merge(channels, ycrcb);
    cv::Mat enhanced;
    cv::cvtColor(ycrcb, enhanced, cv::COLOR_YCrCb2BGR);
    return enhanced;
}

std::vector<Vec4f> detectLineFeatures(Mat gray_image, int line_len)
{
    opts.refine       = 1;  
	opts.scale        = 0.8; 
	opts.sigma_scale  = 1.5;	
	opts.quant        = 2.0;
	opts.ang_th       = 22.5;	
	opts.log_eps      = 0;
	opts.density_th   = 0.6;
	opts.n_bins       = 1024;
	opts.min_length = 0.125;
	cv::Ptr<cv::LineSegmentDetector> ls = cv::createLineSegmentDetector(opts.refine,
	                                                           opts.scale,
	                                                           opts.sigma_scale,
	                                                           opts.quant,
	                                                           opts.ang_th,
	                                                           opts.log_eps,
	                                                           opts.density_th,
	                                                           opts.n_bins);
    std::vector<Vec4f> current_lines;
    std::vector<Vec4f> len_current_lines;
    ls->detect(gray_image, current_lines);
    for (auto line : current_lines) {
        cv::Vec4f cur_line = line;
        float length = cv::norm(cv::Point2f(cur_line[0], cur_line[1]) - cv::Point2f(cur_line[2], cur_line[3]));
        if (length > line_len) {
            len_current_lines.push_back(cur_line);
        }
    }
    return len_current_lines;
}

std::vector<cv::KeyPoint> trackORBFeatures(const cv::Mat gray_image, int point_th) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->setMaxFeatures(point_th);
    orb->setFastThreshold(15);
    orb->setScaleFactor(1.2);

    orb->detectAndCompute(gray_image, cv::noArray(), keypoints, descriptors);

    return keypoints;
}
/**
 * @brief 图像关键点提取和可视化函数 - 从灰度图像中提取线条和关键点特征并进行可视化
 * 
 * 该函数的主要功能：
 * 1. 并行提取线条特征和关键点特征
 * 2. 将特征绘制到彩色图像上进行可视化
 * 3. 返回包含特征信息的图像
 * 
 * 特征类型：
 * - 线条特征：使用LSD(Line Segment Detector)算法检测直线段
 * - 关键点特征：使用ORB算法检测角点和特征点
 * 
 * 可视化效果：
 * - 线条：红色线条，可调节粗细
 * - 关键点：红色矩形框，可调节大小
 * 
 * @param gray_image 输入的灰度图像
 * @param line_th 线条检测阈值（未直接使用，在detectLineFeatures内部使用）
 * @param line_len 最小线条长度阈值
 * @param line_wide 线条绘制宽度
 * @param point_th 关键点数量阈值
 * @param radius_size 关键点绘制半径（转换为矩形边长）
 * @return cv::Mat 包含特征可视化的彩色图像
 */
cv::Mat get_image_keypoints(cv::Mat gray_image, int line_th, int line_len, int line_wide, int point_th, int radius_size){
	// ==================== 变量初始化 ====================
	// 存储检测到的线条特征，每个线条用4个浮点数表示(x1,y1,x2,y2)
	std::vector<cv::Vec4f> lines_ls;
	
	// 存储检测到的关键点特征，包含位置、尺度、方向等信息
	std::vector<cv::KeyPoint> keypoints;
	
	// 将灰度图像转换为BGR彩色图像，用于特征可视化
	// feather_image是全局变量，用于存储最终的可视化结果
	cv::cvtColor(gray_image, feather_image, cv::COLOR_GRAY2BGR);
	
	// ==================== 并行特征提取 ====================
	// 使用Intel TBB并行库同时进行线条和关键点检测
	// 将任务分为2个并行块：线条检测和关键点检测
	tbb::parallel_for(tbb::blocked_range<int>(0, 2), [&](const tbb::blocked_range<int>& r) {
	    // 第一个并行任务：线条特征检测
	    if (r.begin() == 0) {
	        // 使用LSD算法检测直线段特征
	        // detectLineFeatures函数内部会使用line_th等参数
	        lines_ls = detectLineFeatures(gray_image, line_len);
	    } 
	    
	    // 第二个并行任务：关键点特征检测
	    else {
	        // 使用ORB算法检测角点和特征点
	        // trackORBFeatures函数会限制关键点数量为point_th
	        keypoints = trackORBFeatures(gray_image, point_th);
	    }
	});

	// ==================== 特征可视化 ====================
	// 定义线条和关键点的绘制颜色
	cv::Scalar lineColor(55, 55, 255);      // 红色线条 (BGR格式)
	cv::Scalar keypointColor(255, 55, 55);   // 红色关键点 (BGR格式)
	
	// ==================== 绘制线条特征 ====================
	int lineThickness = line_wide;  // 线条粗细
	
	// 遍历所有检测到的线条并绘制
	for (const auto& line : lines_ls) {
	    // 提取线条的起点和终点坐标
	    cv::Point pt1(line[0], line[1]);  // 起点 (x1, y1)
	    cv::Point pt2(line[2], line[3]);  // 终点 (x2, y2)
	    
	    // 在图像上绘制线条
	    cv::line(feather_image, pt1, pt2, lineColor, lineThickness);
	}
	
	// ==================== 绘制关键点特征 ====================
	// 遍历所有检测到的关键点并绘制
	for (const auto& keypoint : keypoints) {
		// 获取关键点的中心坐标
		cv::Point2f point = keypoint.pt;
		
		// 计算矩形框的边长（关键点用矩形框表示）
		int side_length = radius_size * 2;
		
		// 计算矩形框的左上角和右下角坐标
		cv::Point2f upper_left(point.x - side_length / 2, point.y - side_length / 2);
		cv::Point2f bottom_right(point.x + side_length / 2, point.y + side_length / 2);
		
		// 在图像上绘制填充的矩形框表示关键点
		cv::rectangle(feather_image, upper_left, bottom_right, keypointColor, cv::FILLED);
	}
	
	// ==================== 返回结果 ====================
	// 返回包含所有特征可视化的彩色图像
	// 该图像将用于：
	// 1. 调试和可视化特征检测效果
	// 2. 点云着色时的特征匹配
	// 3. 算法性能评估
	return feather_image;
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ConvertToXYZRGBPointCloud(const Vector6dVector& map_cloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (const auto& vec : map_cloud) {
        pcl::PointXYZRGB point;
        point.x = static_cast<float>(vec[0]);
        point.y = static_cast<float>(vec[1]);
        point.z = static_cast<float>(vec[2]);

        uint8_t r = static_cast<uint8_t>(vec[3]);
        uint8_t g = static_cast<uint8_t>(vec[4]);
        uint8_t b = static_cast<uint8_t>(vec[5]);
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                        static_cast<uint32_t>(g) << 8 |
                        static_cast<uint32_t>(b));
        point.rgb = *reinterpret_cast<float*>(&rgb);

        cloud->push_back(point);
    }
    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::RandomSample<pcl::PointXYZRGB> random_sample;
	random_sample.setInputCloud(cloud);
	random_sample.setSample(10000);
	random_sample.filter(*cloud_filtered);
    return cloud_filtered;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "KISS_LV_main");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    KISS_LV * fast_lio_instance = new KISS_LV(nh, nh_private);
    ros::Rate rate(5000);
    ros::spin();
}
