#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
#include <vector>

using namespace Eigen;

class StructParam {
public:
    StructParam() {}
    void setVal(const int x, const int y, const int z, const float param) {
        grid_x_nums = x;
        grid_y_nums = y;
        grid_z_nums = z;
        dyn = param;
    }
    void getVal(int& x, int& y, int& z, float& param) {
        x = grid_x_nums;
        y = grid_y_nums;
        z = grid_z_nums;
        param = dyn;
    }
    
private:
    // x, y, z 方向的栅格数量
    int grid_x_nums{1};
    int grid_y_nums{1};
    int grid_z_nums{1};
    // 动态参数
    float dyn{1.0};
};

class LidarDustRemove {
    public:
    LidarDustRemove() : nh("~") {
        pointCloudSub = nh.subscribe("/base_link_cloud", 2, &LidarDustRemove::process, this);
        gridPointCloudPub = nh.advertise<sensor_msgs::PointCloud2>("/grid_point", 2);
    }

    // 设置每个栅格的三维尺寸
    void SetSize(float x, float y, float z) {
        grid_size_x = x;
        grid_size_y = y;
        grid_size_z = z;
    }

    void process(const sensor_msgs::PointCloud2ConstPtr& input_topic) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*input_topic, *pcl_cloud);
        // 点云 三维栅格化  1. 获取当前点云的三个维度的边界信息  
        // 2. 通过设置的栅格大小将点云进行三个维度的切分  
        // 3.计算切分后的栅格步长
        // 4. 将每个栅格点云填充到对应的map容器里  
        // 5.按照容器ID提取局部点云特征信息
        pcl::PointXYZI minPoint;
        pcl::PointXYZI maxPoint;
        pcl::getMinMax3D(*pcl_cloud, minPoint, maxPoint);

        int xsize = (maxPoint.x - minPoint.x) / grid_size_x + 1;
        int ysize = (maxPoint.y - minPoint.y) / grid_size_y + 1;
        int zsize = (maxPoint.z - minPoint.z) / grid_size_z + 1;

        // 三维栅格
        std::vector<std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>>> blocks(xsize,
         std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>>(ysize,std::vector<pcl::PointCloud<pcl::PointXYZI>>(zsize,
           pcl::PointCloud<pcl::PointXYZI>())));
        
        for (const auto& point : pcl_cloud->points) {
            int x_index = static_cast<int>((point.x - minPoint.x) / grid_size_x);
            int y_index = static_cast<int>((point.y - minPoint.y) / grid_size_y);
            int z_index = static_cast<int>((point.z - minPoint.z) / grid_size_z);

            blocks[x_index][y_index][z_index].points.push_back(point);
        }

        StructParam stParam;
        stParam.setVal(xsize, ysize, zsize, 3.5);
        bool res = AnaysisIntensity(stParam, blocks);
        
        // 部分可视化发布栅格化点云(未被过滤)
        sensor_msgs::PointCloud2 grid_cloud_msg;

        for (const auto& point : pcl_cloud->points) {
            int x_index = static_cast<int>((point.x - minPoint.x) / grid_size_x);
            int y_index = static_cast<int>((point.y - minPoint.y) / grid_size_y);
            int z_index = static_cast<int>((point.z - minPoint.z) / grid_size_z);

            if (blocks[x_index][y_index][z_index].size() > 100 && res != true) {
                pcl::toROSMsg(blocks[x_index][y_index][z_index], grid_cloud_msg);
                break;
            }  
        }

        grid_cloud_msg.header = input_topic->header;
        gridPointCloudPub.publish(grid_cloud_msg);
    }

    private:
    ros::NodeHandle nh;
    ros::Subscriber pointCloudSub;
    ros::Publisher gridPointCloudPub;
    float grid_size_x{0.0};
    float grid_size_y{0.0};
    float grid_size_z{0.0};
    StructParam stParam;

    // 对于每个障碍物物点云栅格化之后 进行栅格特征分析, 是否滤除
    bool AnaysisIntensity(StructParam& dyn,
      const std::vector<std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>>>& blocks) {
        int x_nums, y_nums, z_nums;
        float param;
        float intensity_thd = 15.0;
        int count_nums;
        dyn.getVal(x_nums, y_nums, z_nums, param);
        long total = x_nums * y_nums * z_nums;
        
        for (int i = 0; i < x_nums; i++) {
            for (int j = 0; j < y_nums; j++) {
                for (int k = 0; k < z_nums; k++) {
                    Eigen::Vector4f centroid;
                    if (blocks[i][j][k].size() == 0) {
                        continue;
                    }
                    // 分析每个栅格的特征信息, 这里只分析该障碍物点云中，点云强调的众数，不满足过滤
                    float avg = std::accumulate(blocks[i][j][k].begin(), blocks[i][j][k].end(), 0.0,
                      [&](float sum, pcl::PointXYZI b){return (sum + b.intensity);}) /
                       blocks[i][j][k].size();
                      
                    pcl::compute3DCentroid(blocks[i][j][k], centroid);
                    if (avg > (intensity_thd - (centroid.head(3).norm() / param))) {
                        count_nums++;
                    }
                }
            }
        }
        if (count_nums / total < 0.5) {
            return true;
        }
        return false;
    }
};


int main(int argc, char** args) {
    ros::init(argc, args, "lidar_dust_remove");
    LidarDustRemove gridPartition;
    gridPartition.SetSize(4, 3, 1);
    ros::spin();
    return 0;
}