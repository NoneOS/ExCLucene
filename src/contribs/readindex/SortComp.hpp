
#ifndef SORTCOMP_HPP_
#define SORTCOMP_HPP_

typedef struct _QueryPlan {
    uint32_t termid;
    uint32_t len;
} QueryPlan;

bool compareList(const std::list<uint32_t> &a, const std::list<uint32_t> &b) {
    auto ita = a.begin();
    auto itb = b.begin();
    ++ita;
    ++itb;
    while (ita != a.end() && itb != b.end()) {
        if (*ita < *itb)
            return true;
        else if (*ita > *itb)
            return false;
        ita++;
        itb++;
    }
    if (ita == a.end())
        return true;
    else
        return false;
}

bool ascqLength(const QueryPlan &a, const QueryPlan &b) {
    return a.len < b.len;
}

#endif /* SORTCOMP_HPP_ */
