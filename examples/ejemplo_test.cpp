
#include <nanoflann.hpp>

#include <ctime>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace nanoflann;

void dump_mem_usage();

template <typename T>
struct PointCloud
{
	struct Point
	{
		T  x,y,z;

        Point(double t_x, double t_y, double t_z) : x(t_x), y(t_y), z(t_z) {}
	};

	std::vector<Point>  pts;

	inline size_t kdtree_get_point_count() const { return pts.size(); }

	inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
	{
		const T d0=p1[0]-pts[idx_p2].x;
		const T d1=p1[1]-pts[idx_p2].y;
		const T d2=p1[2]-pts[idx_p2].z;
		return d0*d0+d1*d1+d2*d2;
	}

	inline T kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim==0) return pts[idx].x;
		else if (dim==1) return pts[idx].y;
		else return pts[idx].z;
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};


int main()
{
    //std::vector<Point> points = {
        //Point(0,0,0),
        //Point(5,0,0),
        //Point(-3,0,0),
        //Point(3,3,0),
    //};

	PointCloud<double> cloud;

    cloud.pts.emplace_back(0.0, 0.0, 0.0);
    cloud.pts.emplace_back(5.0, 0.0, 0.0);
    cloud.pts.emplace_back(-3.0, 0.0, 0.0);
    cloud.pts.emplace_back(3.0, 3.0, 0.0);

	double query_pt[3] = { 1.0, 1.0, 1.0};

	typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<double, PointCloud<double> > ,
		PointCloud<double>,
		3> my_kd_tree_t;

	my_kd_tree_t index(3, cloud, KDTreeSingleIndexAdaptorParams(1));
	index.buildIndex();

    size_t num_results = 5;
    std::vector<size_t>   ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);
    num_results = index.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);

    ret_index.resize(num_results);
    out_dist_sqr.resize(num_results);

    cout << "knnSearch(): num_results=" << num_results << "\n";
    for (size_t i=0;i<num_results;i++)
        cout << "idx["<< i << "]=" << ret_index[i] << " dist["<< i << "]=" << out_dist_sqr[i] << endl;
    cout << "\n";
	return 0;
}

