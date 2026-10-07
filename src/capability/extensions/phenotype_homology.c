/* Exact BLOSUM62 Smith-Waterman carry; no packed coordinates or length cutoff.
 * First M/X/Y predecessor wins ties; gap opening wins extension ties;
 * first row-major positive match-state optimum wins. Gap cost = 11 + k.
 * Scores are integers because the canonical matrix and gap costs are integers.
 */
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

typedef struct { int score, columns, identical, paired, start_a, start_b; } State;
static State negative(void) { State s = {-1000000000, 0, 0, 0, 0, 0}; return s; }
int phenotype_align(const unsigned char *a, int n, const unsigned char *b, int m,
                    const int32_t *table, double *out) {
    State *storage = malloc(6 * ((size_t)m + 1) * sizeof(State));
    if (!storage) return 1;
    State *pm=storage, *px=pm+m+1, *py=px+m+1;
    State *cm=py+m+1, *cx=cm+m+1, *cy=cx+m+1;
    for (size_t k=0; k<6*((size_t)m+1); ++k) storage[k]=negative();
    State best=negative(); int end_a=0, end_b=0;
    for (int i=1; i<=n; ++i) {
        cm[0]=cx[0]=cy[0]=negative();
        for (int j=1; j<=m; ++j) {
            State s=pm[j-1];
            if (px[j-1].score>s.score) s=px[j-1];
            if (py[j-1].score>s.score) s=py[j-1];
            if (s.score<=0) { s=negative(); s.score=0; s.start_a=i; s.start_b=j; }
            s.score+=table[20*a[i-1]+b[j-1]];
            s.columns++; s.paired++; s.identical+=(a[i-1]==b[j-1]); cm[j]=s;
            State opened=pm[j], extended=px[j]; opened.score-=12; extended.score-=1;
            s=extended.score>opened.score?extended:opened; s.columns++; cx[j]=s;
            opened=cm[j-1]; extended=cy[j-1]; opened.score-=12; extended.score-=1;
            s=extended.score>opened.score?extended:opened; s.columns++; cy[j]=s;
            if (cm[j].score>best.score) { best=cm[j]; end_a=i; end_b=j; }
        }
        State *t=pm; pm=cm; cm=t; t=px; px=cx; cx=t; t=py; py=cy; cy=t;
    }
    for (int k=0; k<8; ++k) out[k]=0;
    if (best.score>0) {
        out[0]=best.score; out[1]=best.columns; out[2]=best.identical;
        out[3]=100.0*(end_a-best.start_a+1)/n;
        out[4]=100.0*(end_b-best.start_b+1)/m;
        out[5]=best.paired; out[6]=end_a-best.start_a+1; out[7]=end_b-best.start_b+1;
    }
    free(storage); return 0;
}
