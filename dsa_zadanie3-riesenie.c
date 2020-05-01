// zadanie3.c -- Zadanie 3 - Popolvar
// Peter Markus, 28.11.2016 10:04:35

#include <stdio.h>
#include <stdlib.h>
#define INF 100000

typedef struct edge {
    int pos;
    int length;     // dlzka prechodu
    struct edge *next;
}EDGE;

typedef struct heapNode {
    int sDist;             // dlzka cesty od startu
    int pos;               // pozicia 2d vyjadrena cislom
}HEAPNODE;

typedef struct priorFront{
    HEAPNODE data[INF];
    int pt;
}PRIORFRONT;

PRIORFRONT gFront;
int xyAmount = 0;

int getLength(char c) {
    switch(c) {
        case 'N': return INF; break;
        case 'H': return 2; break;
        default : return 1; break;
    }
}

void push(HEAPNODE node)   // vkladanie vrcholu do haldy
{
    HEAPNODE tmp;
    int pt = gFront.pt;
    gFront.data[gFront.pt++] = node;

    for( ; pt >= 1; pt = (pt-1)/2)  // ak [pt]child je mensi ako [(pt-1)/2]parent -> vymena
        if(gFront.data[pt].sDist < gFront.data[(pt-1)/2].sDist){
            tmp = gFront.data[pt];
            gFront.data[pt] = gFront.data[(pt-1)/2];
            gFront.data[(pt-1)/2] = tmp;
        }
}

HEAPNODE pop()  // vyberanie vrcholu z haldy
{
    HEAPNODE tmp, p1, p2, top = gFront.data[0];   // zobere top
    gFront.data[0] = gFront.data[--gFront.pt];    // na top dame posledne pushnuty
    int n = 0;

    while(n <= gFront.pt){
        p1.sDist = p2.sDist = INF;
        if(n*2+1 < gFront.pt && gFront.data[n*2+1].sDist < gFront.data[n].sDist)  //indexy && vzdialenosti
            p1 = gFront.data[n*2+1];    // lavy detsky kandidat
        if(n*2+2 < gFront.pt && gFront.data[n*2+2].sDist < gFront.data[n].sDist)
            p2 = gFront.data[n*2+2];    // pravy detsky kandidat
        if(p1.sDist == INF && p2.sDist == INF)
            break;
        if(p1.sDist < p2.sDist){    // porovnanie potencionalnych deti
            tmp = gFront.data[n*2+1];
            gFront.data[n*2+1] = gFront.data[n];
            gFront.data[n] = tmp;                    // vymena rodica s lavym synom
            n = n*2+1;
        }
        else {                      // vymena rodica s pravym synom
            tmp = gFront.data[n*2+2];
            gFront.data[n*2+2] = gFront.data[n];
            gFront.data[n] = tmp;                    // vymena rodica s pravym synom
            n = n*2+2;
        }
    }
    return top;
}

void backward(int dest, int m, int path[], int *output)
{
    if(path[dest] != -1)                        // kontrola ci niesme na zaciatku PO
        backward(path[dest], m, path, output);  // iterujeme dozadu
    output[xyAmount++] = dest % m;              // suradnica y
    output[xyAmount++] = dest / m;              // suradnica x
}

void forward(int dest, int m, int path[], int pom, int *output)
{
    if(pom != 0){                       // len pri prvom volani funkcie vynechame PR
        output[xyAmount++] = dest % m;  // suradnica y
        output[xyAmount++] = dest / m;  // suradnica x
    }
    if(path[dest] != -1)
        forward(path[dest], m, path, ++pom, output);    // iterujeme dopredu
}

int *dijkstra(EDGE **graph, int start, int n, int m, int *path)
{
    // inicializacia
    HEAPNODE node;
    EDGE *te;   // temporary edge
    int i, pos, nodeAmount = m*n;
    int *sDist  = (int *)malloc(nodeAmount*sizeof(int));

    for(i = 0; i < nodeAmount; i++){
        sDist[i] = INF;     // hodnota - vzdialenost medzi poziciou startu a poziciou i
        path[i]  = -1;      // hodnota - najvyhodnejsia pozicia z ktorej sa ide do pozicie i(index)
    }
    sDist[start] = 0;
    gFront.pt    = 0;
    node.sDist   = 0;
    node.pos     = start;
    push(node);

    for(i = 0; i < nodeAmount; i++) {
        node = pop();
        pos = node.pos;   // zaciname od pozicie top vrcholu (s najmensou sDist)

        for(te = graph[pos]; te != NULL; te = te->next)     // ideme po vsetkych susedoch vrcholu
            if(sDist[pos] + te->length < sDist[te->pos]){   // kratsia cesta najdena (akt.cesta  + prech.cest < uzVyp.cesta)
                path[te->pos]  = pos;
                sDist[te->pos] = sDist[pos] + te->length;
                node.sDist     = sDist[pos] + te->length;   // ide sa ulozit vrchol do haldy
                node.pos       = te->pos;
                push(node);
             }
     }
     return sDist;
}

EDGE *insert(EDGE *actual, int pos, int length)
{
    EDGE *addPrev   = (EDGE *)calloc(1,sizeof(EDGE));
    addPrev->length = length;
    addPrev->pos   = pos;
    addPrev->next   = actual;
    return addPrev;
}

int *zachran_princezne(char **mapa, int n, int m, int t, int *dlzka_cesty)
{
    // inicializacia
    EDGE **graph = (EDGE **)calloc(n*m, sizeof(EDGE *));
    int i, j, k, length, sum, min = INF;
    int data[3], dragon, princess[3], pi = 0;       // princess iterator
    int *output = (int *) malloc(INF*sizeof(int));
    int **dista = (int **)malloc(3*sizeof(int *));  // pole poli so vzdialenostami roznych startov
    int **path  = (int **)malloc(4*sizeof(int *));  // pole poli s najvyhodnejsimi CP z ktoreho sa prechadza do CP v druhom indexe
    for(i = 0; i < 4; i++)
        path[i] = (int *)malloc(m*n*sizeof(int));
    xyAmount = 0;

    // vytvaranie grafu z 2d mapy charov    (graf je predstaveny polom struktur EDGE)
    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            if(mapa[i][j] == 'D')
                dragon = i*m + j;
            if(mapa[i][j] == 'P')
                princess[pi++] = i*m + j;   // i*m + j = cislo vyjadrujuce poziciu v 2d mape (CP)

            length = getLength(mapa[i][j]);
            if(i > 0)   // pridavanie vsetkych susednych vrcholov cez spajany zoznamum do grafu
                graph[i*m + j] = insert(graph[i*m + j], (i-1)*m + j, length);
            if(j > 0)
                graph[i*m + j] = insert(graph[i*m + j], i*m + j-1, length);
            if(j < m-1)
                graph[i*m + j] = insert(graph[i*m + j], i*m + j+1, length);
            if(i < n-1)
                graph[i*m + j] = insert(graph[i*m + j], (i+1)*m + j, length);
        }
    }

    // hladanie najkratsich ciest
                dijkstra(graph,           0, n, m, path[0]); // nemusime ratat vzdialenost k drakovi, staci path
    dista[0]  = dijkstra(graph, princess[0], n, m, path[1]); // vraciam pole vzdialenostii ku vsetkym bodom od princeznej
    dista[1]  = dijkstra(graph, princess[1], n, m, path[2]); // path - cesta ku kazdemu bodu
    dista[2]  = dijkstra(graph, princess[2], n, m, path[3]);

    // hladanie najkratsej kombinacie ciest
    for(i = 0; i < 3; i++)
        for(j = 0; j < 3; j++)
            for(k = 0; k < 3; k++)
                if((i != j) && (i != k) && (j != k)){
                    sum = dista[i][dragon]      +
                          dista[j][princess[i]] +
                          dista[k][princess[j]];
                    if(min >= sum){
                        min = sum;
                        data[0] = i;
                        data[1] = j;
                        data[2] = k;
                    }
                }

    // ukladanie postupnosti krokov po mape do outputu
    backward(          dragon, m, path[0], output);
    forward(           dragon, m, path[data[0]+1], 0, output);        // k prvej najblizsej princzn od draka, +1 tam je kvoli tomu ze path[0] bol urceny pre draka
    forward(princess[data[0]], m, path[data[1]+1], 0, output);
    forward(princess[data[1]], m, path[data[2]+1], 0, output);
    *dlzka_cesty = xyAmount / 2;

    return output;
}

char **createMap(char **map) {

    int i, j, m, n, t;
    FILE *f;

    char c = NULL;
    f = fopen("mapa2.txt", "r");
    fscanf(f, "%d %d %d\n", &n, &m, &t);

    map = (char **)malloc(n * sizeof(char *));
    for (i = 0; i < n; i++)
         map[i] = (char *)malloc(m * sizeof(char));

    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++) {
            c = fgetc(f);
            map[i][j] = c;
        }

    fclose(f);
    return map;
}

int main() {
    int dlzka_cesty, i;
    char **mapa = NULL;

    mapa = createMap(mapa);

    int *cesta = zachran_princezne(mapa, 5, 5, 30, &dlzka_cesty);

    for(i = 0; i < dlzka_cesty; ++i)
        printf("%d.\t%d %d\n", i, cesta[i*2], cesta[i*2+1]);

    return 0;
}

/*
    int lo, ite = 0;
    printf("____CESTICKA______\n");
    for(lo = 0; lo < 75; lo+= 2, ite++){
        printf("%d.\tx:%d y:%d\n", ite, vystup[lo], vystup[lo+1]);
    }
    printf("\nlength CESTY: %d\n", *length_cesty);

    printf("\nGRAF\n");
    int la, lo;
    for(lo = 0; lo < n; lo++) {
        for(la = 0; la < m; la++) {
            printf("%c", mapa[lo][la]);
        }
        printf("\n");
    }
*/
