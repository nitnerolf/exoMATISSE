/***************************************************************************
 *                                                                         *
 *                              fitOmatic                                  *
 *                   Model-fitting prototyping utility                     *
 *                                                                         *
 *                      Copyright 2007, F. Millour                         *
 *                            fmillour@oca.eu                              *
 *                                                                         *
 ***************************************************************************
 *
 * Test of observation simulation tools
 *
 * Please note that this script is distributed under the GPL licence,
 * available at http://www.gnu.org/licenses/gpl.txt
 *
 * Please ACKNOWLEDGE the use of this script for any use in a publication
 *
 ***************************************************************************
 *
 * "@(#) $Id: testSimuInterf.i 631 2015-11-24 18:10:52Z fmillour $"
 *
 ***************************************************************************/

include,"fitOmatic.i";
wkll;

loca = "calern"; //loc = "e-elti";
/************************************************************************/

stations = stats = [];
stats = _simiSaveStations(loca);
N = numberof(stats);
astats = where(strmatch(stats,"A"));
Na = numberof(astats);



// Configs P87
stations = [
            ////////////////////////
            ["A1", "B2", "C1"],
            // ["A1", "B2", "D0"],
            // ["A1", "C1", "D0"],
            ["B2", "C1", "D0"],
            ///////////////////////
            ["A1", "K0", "G1"],
            // ["A1", "K0", "I1"],
            // ["A1", "G1", "I1"],
            ["K0", "G1", "I1"],
            ///////////////////////
            ["D0", "H0", "G1"],
            ["D0", "H0", "I1"]// ,
            // ["D0", "G1", "I1"],
            // ["H0", "G1", "I1"]
            ///////////////////////
            ];




//stations = simiGetNonRedundantStations();

cen = 2.16555e-6;
bw = 0.000000086e-6;
NN = 2;
R = cen/bw*NN;
write,"Resolution",R;
lambda = span(cen-bw/2,cen+bw/2,NN);
bandwidth = lambda/R;
nlambda = numberof(lambda);
nObs = dimsof(stations)(0);
maxBase = 150;

yocoNmCreate,2,2,2,square=1,width=850,height=1000,fx=1,fy=1, dx=0.1, dy=0.06;


col = ["red", "green", "blue", "cyan", "magenta", "yellow", "black"];
for(kObs=1;kObs<=nObs;kObs++)
    //kObs=1;
{
    /*******************************************************/
    // Stations
    plsys,1;
    //fma;
    simiPlotStations;
    simiPlotUsedStations,"", stations(,kObs);

    
    /*******************************************************/
    // UV coverage
    
    plsys,3;
    dec = -60;
    stardata = simiSTARVIS(date="01/10/2008",
                           ra=0.0,
                           dec=dec);
    
    
    xyTitles = ["E <------ Proj. baseline (m)",
                "Proj. baseline (m) ------> N"];
    
    hastep  = 0.1;
    harange = 8.0;
    coloring = lst = uvTable = [];
    uvwTable = simiPlotUV(stardata,stations(,kObs),lst,coloring,
                          lambda=1,
                          mx=maxBase,
                          hacen=0,
                          hastep=hastep,
                          fp=0,
                          colors="bases",
                          frac=0.0,
                          msize=4,
                          marker='\1',
                          xyTitles=xyTitles,
                          spFreqFact=,
                          harange=harange,
                          place=loca);
    limits,maxBase,-maxBase,-maxBase,maxBase
    
    
        /*******************************************************/
        // UV coverage 2

        plsys,4;

    u = uvwTable(1,..);
    v = uvwTable(2,..);
    BL = abs(u,v);
    PA = 180/pi*atan(u,v);

    nbBases = dimsof(BL)(2);
    for(iBase=1;iBase<=nbBases;iBase++)
    {
        Phases = array(5./6.*2*pi*(iBase-1)/double(nbBases-1),
                       nLambda);
        kolor = get_color(1,Phases(1));

        plg, BL(iBase,,1), PA(iBase,,1), color=kolor, width=4,type="none",marker='\1', width=100;
        plg, BL(iBase,,1), PA(iBase,,1)+180, color=kolor, width=4,type="none",marker='\1', width=100;
        plg, BL(iBase,,1), PA(iBase,,1)-180, color=kolor, width=4,type="none",marker='\1', width=100;
    }
    limits,0,180,0,maxBase;
    xytitles,"Position angle (^o^)","Projected baseline (m)",[0,0.02];
    
    
    /*******************************************************/
    // Delay

    plsys,2;
    pltitle,strpart(sum(stations(,kObs)+"-"),:-1)
        w = uvwTable(3,..);
    ha = lst(,1)+stardata.ra;

    for(iBase=1;iBase<=nbBases;iBase++)
    {
        Phases = array(5./6.*2*pi*(iBase-1)/double(nbBases-1),
                       nLambda);
        kolor = get_color(1,Phases(1));
        plg, w(iBase,,1), ha, color=kolor, width=4,type="none",marker='\1';
    }
    limits,,,-maxBase,maxBase;
    xytitles,"Hour angle","Delay (m)",[0,0.02];
    pldj,-4,100,4,100, type="dash";
    pldj,-4,-100,4,-100, type="dash";


    hcps,"~/UV_"+strpart(sum(stations(,kObs)+"-"),:-1)+"_"+pr1(dec)+"_"+"Stations.ps";
    require,"chg_bbox.i";
    chg_bbox,"~/UV_"+strpart(sum(stations(,kObs)+"-"),:-1)+"_"+pr1(dec)+"_"+"Stations.ps";
    
}

