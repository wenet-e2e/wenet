#pragma once
#include <vector>
#include <unordered_map>
using namespace std;

namespace victor {

    class Node
    {
    public:
        int w;   //the symbol at this node

        int idx; //the index of this node
        Node* fater_node, * fail_node;
        double fail_cost;
        vector<Node*> next_nodes;
        vector<double> next_nodes_cost;
        unordered_map<int, pair<double, int>> outputw_2_scrore_next_states;
        bool next_node(int w, Node** next_node, double* cost);
        // return wether w is in the next node of current node.
        //if true change the two pointer to point at the next node ant its cost
        //if not change the two pointer to point at the fail node ant fail cost

    };



    struct Score_output
    {

        unordered_map<int, pair<double, int>>* outputw_2_scrore_next_states;
        double return_cost;     //the score of the rest of the symbols.
    };


    class HWScorer
    {

    public:

        Node root_node;
        HWScorer(vector<vector<int>> hotwords,double weight);
        int init_state(void);
        Score_output score(int state_int);

        vector<Node*> all_nodes;

    private:
        void add_hotword(vector<int> hotword);
        double gen_score(int i);//generate the award at i th symbol.
        double weight_ = 1;
        void BFS_re_route_failpath(void);  //go through all node with BFS and reroute all the fail path
        void re_route_failpath(Node* node);
        void optimize_node(Node* node); //optimize this node

    };



}