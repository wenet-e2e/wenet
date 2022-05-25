#include "decoder/hw_scorer.h"
#include <iostream>

using namespace std;
namespace victor {

    template <typename content>
    bool vector_contain(vector<content> vec, content w)
    {
        for (int i = 0; i < vec.size(); i++)
        {

            if (vec[i] == w)
                return true;

        }
        return false;
    }




    bool Node::next_node(int w, Node** next_node, double* cost)
    {
        for (int i = 0; i < next_nodes.size(); i++)
        {
            if (next_nodes[i]->w == w)
            {
                *next_node = next_nodes[i];
                *cost = next_nodes_cost[i];
                return true;
            }

        }
        *next_node = fail_node;
        *cost = fail_cost;
        return false;


    }




    HWScorer::HWScorer(vector<vector<int>> hotwords,double weight)
    {
        weight_ = weight;
        root_node.fail_cost = 0;
        root_node.fail_node = &root_node;
        root_node.idx = 0;

        all_nodes.push_back(&root_node);

        for (int i = 0; i < hotwords.size(); i++)
        {
            add_hotword(hotwords[i]);
        }



        BFS_re_route_failpath();

        for (int i = 0; i < all_nodes.size(); i++)
        {
            optimize_node(all_nodes[i]);
        }


    }






    Score_output HWScorer::score(int cur_state_int)
    {

        Node* node_p = all_nodes[cur_state_int];
        Score_output output;


        output.outputw_2_scrore_next_states = &node_p->outputw_2_scrore_next_states;


        output.return_cost = node_p->fail_cost;

        return output;
    }


    void HWScorer::optimize_node(Node* node)
    {


        vector<int> next_w;
        double total_fail_cost = 0;
        double temp_cost = 0;
        for (int i = 0; i < node->next_nodes.size(); i++)
        {
            next_w.push_back(node->next_nodes[i]->w);
            node->outputw_2_scrore_next_states[node->next_nodes[i]->w] = make_pair(node->next_nodes_cost[i], node->next_nodes[i]->idx);
        }
        Node* node_p = node;

        //vector_contain
        while (node_p != &root_node)
        {
            total_fail_cost += node_p->fail_cost;
            node_p = node_p->fail_node;
            for (int i = 0; i < node_p->next_nodes.size(); i++)
            {
                if (!vector_contain(next_w, node_p->next_nodes[i]->w))
                {
                    temp_cost = total_fail_cost + node_p->next_nodes_cost[i];
                    next_w.push_back(node_p->next_nodes[i]->w);
                    node->next_nodes.push_back(node_p->next_nodes[i]);
                    node->next_nodes_cost.push_back(temp_cost);
                    node->outputw_2_scrore_next_states[node_p->next_nodes[i]->w] = make_pair(temp_cost, node_p->next_nodes[i]->idx);


                }
            }


        }
        node->fail_cost = total_fail_cost;
        node->fail_node = &root_node;


    }


    void HWScorer::BFS_re_route_failpath(void)
    {
        vector<Node*> children_nodes = { &root_node };
        vector<Node*> temp_children_nodes = { };

        while (children_nodes.size() != 0)
        {
            temp_children_nodes = { };
            for (int i = 0; i < children_nodes.size(); i++)
            {
                for (int j = 0; j < (children_nodes[i]->next_nodes).size(); j++)
                {
                    re_route_failpath(children_nodes[i]->next_nodes[j]);
                    temp_children_nodes.push_back(children_nodes[i]->next_nodes[j]);

                }

            }
            children_nodes = temp_children_nodes;
        }



    }


    void HWScorer::re_route_failpath(Node* node)
    {


        if (node->fail_cost == 0 and node->fail_node == &root_node)
            return; //this is a terminal node don't reroute its fail path



        int w = node->w;
        Node* node_p = node->fater_node;
        Node* next_node;
        double total_return_cost = 0;
        double current_cost = 0;

        node_p->next_node(w, &next_node, &current_cost);
        total_return_cost = -current_cost;

        while (node_p != &root_node)
        {
            total_return_cost += node_p->fail_cost;

            if (node_p->fail_node->next_node(w, &next_node, &current_cost))
            {
                node->fail_node = next_node;
                node->fail_cost = total_return_cost + current_cost;
                return;
            }
            else
            {
                node_p = node_p->fail_node;
            }

        }
        node->fail_cost = total_return_cost;
        node->fail_node = &root_node;        //no match just return to root node
    }

    void HWScorer::add_hotword(vector<int> hotword)
    {
        Node* node_p = &root_node;
        int w = 0;
        double cost;
        Node* next_node_p;
        bool is_match;
        double award;
        for (int i = 0; i < hotword.size(); i++)
        {

            award = gen_score(i);
            w = hotword[i];
            is_match = node_p->next_node(w, &next_node_p, &cost);

            if (is_match)
            {
                node_p = next_node_p;
            }
            else
            {
                Node* new_node_p = new Node;
                new_node_p->idx = all_nodes.size();
                all_nodes.push_back(new_node_p);

                new_node_p->w = w;
                new_node_p->fater_node = node_p;
                node_p->next_nodes.push_back(new_node_p);
                node_p->next_nodes_cost.push_back(award);
                node_p = new_node_p;

            }

        }

        node_p->fail_node = &root_node;
        node_p->fail_cost = 0;




    }


    double HWScorer::gen_score(int i)
    {


        return weight_;

    }

    int HWScorer::init_state(void)
    {

        return 0;
    }


}