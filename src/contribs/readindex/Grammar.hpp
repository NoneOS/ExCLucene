#ifndef GRAMMAR_HPP_
#define GRAMMAR_HPP_

/***********************Constraints****************************/
namespace Grammar_Consts {

    const uint32_t pp_max_productions = 18590687;    //gov2 (7032739)
    //const uint32_t pp_max_productions = 40103280; //twitter (40103280)
    const uint32_t rp_max_productions = 145000033;

    const uint32_t hash_tbl_size = 145000043;
//    const uint32_t hash_tbl_size = 154412153;
    const uint32_t digram_index_size = 128010167;

    const uint32_t REDUCED_PRODUCTION = 0x40000000;
    const uint32_t CONTAIN_NT = 0x80000000;
    const uint32_t PRODUCTION_MASK = 0x3FFFFFFF;
    const uint32_t NT = 0x80000000;

    const uint32_t docid_max = 25205178;            //gov2
    //const uint32_t docid_max = 40103280; //twitter
    //const uint32_t list_upperb = 40200000;
    const uint32_t list_upperb = 26000000;

    enum PostingFormat {
        TwoPart = 0, FourPart, OddEven
    }; //Postings fileFormat

    const uint32_t nbflag = 1u << 22;
}

/***********************Production****************************/
typedef struct Node {
    uint32_t num;
    struct Node *before;
    struct Node *next;
} Node;

typedef struct Production {
    Node head;
    Node tail;
    uint32_t len;
    uint32_t sn;
    bool rdc;
    struct Production *next;
} Production;

typedef struct Digram {
    Node *index;
    Production *pro;
    struct Digram *next;
} Digram;

inline uint32_t get_rpnum(const uint32_t symb) {
    return (symb & Grammar_Consts::PRODUCTION_MASK);
}

inline bool is_NT(const uint32_t symb) {
    return (symb & Grammar_Consts::NT);
}

// two-part compressed files: lid = lid << 2 | 0x2 |0x1

inline bool contain_NT(const uint32_t lid) {
    return (lid & 0x2);
}

inline bool contain_TT(const uint32_t lid) {
    return (lid & 0x1);
}

inline bool chknsame(const uint32_t a, const uint32_t b) {
    return ((a ^ b) & 0x1);
}

/*********************Statement Only**************************/
void Option_Analy(const int argc, const char* argv[]);

#endif /* GRAMMAR_HPP_ */

