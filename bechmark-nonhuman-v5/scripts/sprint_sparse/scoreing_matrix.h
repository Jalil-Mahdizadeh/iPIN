#ifndef SCOREING_MATRIX_H_
#define SCOREING_MATRIX_H_
#include "global_parameters.h"
#include "PtoHSP.h"
#include <unordered_map>

// Same serial SPRINT arithmetic and per-cell accumulation order. Native HSP
// preprocessing and sorting happen before this class is constructed. Cells
// outside the requested pair list cannot affect a requested score.
class SCORING_MATRIX {
public:
    vector<pair<int,int> > pairs;
    vector<float> scores;
    unordered_map<uint64_t,size_t> cells;
    vector<vector<pair<int,size_t> > > neighbors;
    vector<unordered_map<int,vector<int> > > positions;
    static uint64_t key(int a,int b) {
        if(a>b) swap(a,b);
        return (uint64_t(uint32_t(a))<<32)|uint32_t(b);
    }
    SCORING_MATRIX(PtoHSP &hsp): neighbors(num_protein),positions(num_protein) {
        const string files[2]={TEST_FN_POS,TEST_FN_NEG};
        for(int f=0;f<2;++f) {
            ifstream in(files[f].c_str());if(!in) exit(65);
            string aa,bb;
            while(in>>aa>>bb) {
                int a=p_name_id.at(aa),b=p_name_id.at(bb);uint64_t k=key(a,b);
                if(cells.count(k)) continue;
                size_t c=pairs.size();cells[k]=c;pairs.push_back(make_pair(a,b));scores.push_back(0.f);
                neighbors[a].push_back(make_pair(b,c));
                if(a!=b) neighbors[b].push_back(make_pair(a,c));
            }
        }
        for(int p=0;p<num_protein;++p)
            for(int j=0;j<(int)hsp.HSP_table[p].size();++j)
                positions[p][hsp.HSP_table[p][j].p2_id].push_back(j);
    }
    void load_traing(string filename,PtoHSP &hsp) {
        ifstream in(filename.c_str());if(!in) exit(3);
        string aa,bb;size_t count=0;uint64_t updates=0;
        while(in>>aa>>bb) {
            if(!p_name_id.count(aa)||!p_name_id.count(bb)) {cerr<<"Unknown training protein"<<endl;exit(66);}
            int a=p_name_id.at(aa),b=p_name_id.at(bb);if(b>=a) swap(a,b);
            const vector<HSP_OCC>& left=hsp.HSP_table[a];
            const vector<HSP_OCC>& right=hsp.HSP_table[b];
            for(int i=0;i<(int)left.size();++i) {
                const HSP_OCC &x=left[i];
                for(const auto &neighbor:neighbors[x.p2_id]) {
                    auto found=positions[b].find(neighbor.first);
                    if(found==positions[b].end()) continue;
                    // For a fixed i each destination is a separate score cell;
                    // every j contributing to that cell retains native order.
                    for(int j:found->second) {
                        if(a==b && j<i) continue;
                        const HSP_OCC &y=right[j];
                        scores[neighbor.second] += (x.hsp_calculation_score * (y.length-k_size+1) + y.hsp_calculation_score * (x.length-k_size+1));
                        ++updates;
                    }
                }
            }
            if(++count%10000==0) cout<<"Training pairs "<<count<<" score updates "<<updates<<endl;
        }
        for(size_t c=0;c<pairs.size();++c) {
            int a=pairs[c].first,b=pairs[c].second;
            scores[c]=scores[c]/((float)p_id_seq.at(a).length()*(float)p_id_seq.at(b).length());
        }
        cout<<"Training pairs "<<count<<" score updates "<<updates<<" requested pairs "<<pairs.size()<<endl;
    }
    void load_test(string filename,char flag) {
        ifstream in(filename.c_str());if(!in) exit(3);
        ofstream out(OUTPUT_FN.c_str(),ios::app);
        ofstream split((flag=='p'?OUTPUT_pos_FN:OUTPUT_neg_FN).c_str(),ios::app);
        string aa,bb;
        while(in>>aa>>bb) {
            float value=scores.at(cells.at(key(p_name_id.at(aa),p_name_id.at(bb))));
            int label=flag=='p'?1:0;out<<value<<" "<<label<<"\n";split<<value<<" "<<label<<"\n";
        }
    }
    void print_entire_final_score_matrix() { cerr<<"Explicit candidates required"<<endl;exit(64); }
};
#endif
