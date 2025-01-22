/***************************************************************************
 *                                                                         *
 *                              fitOmatic                                  *
 *                   Model-fitting prototyping utility                     *
 *                                                                         *
 *                    Copyright 2007-2011, F. Millour                      *
 *                            fmillour@oca.eu                              *
 *                                                                         *
 ***************************************************************************
 *
 * Simulation tools for interferometric arrays
 *
 * Please note that this script is distributed under the GPL licence,
 * available at http://www.gnu.org/licenses/gpl.txt
 *
 * Please ACKNOWLEDGE the use of this script for any use in a publication
 *
 ***************************************************************************
 *
 * "$Id: simuInterf.i 627 2015-11-24 17:57:50Z fmillour $"
 *
 ***************************************************************************/

func simuInterf(void)
/* DOCUMENT simuInterf

       DESCRIPTION
       Simulation tools for interferometric arrays
       
       VERSION
       $Revision: 627 $

       REQUIRE
       fitOmatic.i
       yoco packages (yoco.i)

       CAUTIONS

       AUTHORS
       See AUTHORS file at the root of the directory distribution

       CONTRIBUTIONS

       FUNCTIONS
   - getColVal                  : 
   - simiComputUvCoord          : 
   - simiComputeBaseVect        : 
   - simiGenerateUV             : 
   - simiGetBaseVect            : 
   - simiGetLocation            : 
   - simiGetNonRedundantStations: 
   - simiGetUVCoordinates       : 
   - simiPlotStations           : 
   - simiPlotUV                 : 
   - simiPlotUVCoordinates      : 
   - simiPlotUsedStations       : 
   - simuInterf                 : 
    */
{
    version = strpart(strtok("$Revision: 627 $",":")(2),2:-2);
    if (am_subroutine())
    {
        help, simuInterf;
    }
    return version;
} 

yocoLogInfo,"#include \"simuInterf.i\"";


require,"visModels.i";
require,"photocentreModels.i"; 


/************************************************************************/ 

func simiGenerateUV(stardata, stations, &lst, &coloring, &usedStations, hacen=, harange=, hastep=, lambda=, colors=, mx=, fp=, frac=, msize=, marker=, spFreqFact=, xyTitles=, place=, interferometer=)
/* DOCUMENT simiGenerateUV(stardata, stations, &lst, &coloring, &usedStations, hacen=, harange=, hastep=, lambda=, colors=, mx=, fp=, frac=, msize=, marker=, spFreqFact=, xyTitles=, place=, interferometer=)

       DESCRIPTION

       PARAMETERS
   - stardata      : 
   - stations      : 
   - lst           : 
   - coloring      : 
   - usedStations  : 
   - hacen         : 
   - harange       : 
   - hastep        : 
   - lambda        : 
   - colors        : 
   - mx            : 
   - fp            : 
   - frac          : 
   - msize         : 
   - marker        : 
   - spFreqFact    : 
   - xyTitles      : 
   - place         : 
   - interferometer: 

       RETURN VALUES

       CAUTIONS

       EXAMPLES

       SEE ALSO
    */
{
    if(is_void(place))
    {
        place="paranal";
        if(is_void(interferometer))
            interferometer="VLTI";
    }
    
    simiGetLocation, place, interferometer, loc;

    if(is_void(lambda))
    {
        nLambda = 20;
        lambdaMin = 1.1 * micro2unit;
        lambdaMax = 2.5 * micro2unit;
        lambda = span(lambdaMax,lambdaMin,nLambda);
    }
    else
        nLambda = numberof(lambda);

    if(is_void(colors))
        colors="wlen";

    //colors="bases";
    // colors="obs";

    if(is_void(spFreqFact))
        spFreqFact = 1.0;

    if(is_void(xyTitles))
        xyTitles = ["E <------ Spatial frequency (m^-1^)",
                    "Spatial frequency (m^-1^) ------> N"];

    if(is_void(frac))
        frac = 1.0;

    if(is_void(msize))
        msize = 1.0;

    if(is_void(marker))
        marker = '\1';

    if(is_void(hacen))
        hacen = stardata.ra;
    if(is_void(harange))
        harange = 8.0;

    hamin = hacen-harange/2;
    hamax = hacen+harange/2;

    if(is_void(hastep))
        hastep = 70./60.;

    dimz = dimsof(stations);
    if(dimz(1)==1)
        nObs = 1;
    if(dimz(1)==2)
        nObs = dimz(0);

    uvwTable = lst = coloring = [];

    usedStations = [];
    for(kObs=1;kObs<=nObs;kObs++)
    {
        nbTel = numberof(stations(,kObs));
        nbBases = nbTel*(nbTel-1)/2;

        baseNames = B = orig = [];

        getPhase = [0.0,2*pi/3,4*pi/3,pi,5*pi/3.0,pi/3,0.0];
        getIntens = [1.0,1.0,1.0,2.0,2.0,2.0,0.0];

        uSt = [];
        B = simiComputeBaseVect(orig, stations(,kObs), baseNames, fixDelay,uSt);
        
        u = v = w = theta = base = [];

        for(iObs=hamin;iObs<=hamax;iObs+=hastep)
        {
            grow,usedStations,[uSt];
            
            stardata.lst = double(iObs)+hastep/2.0*frac*random_n();
            grow,lst,stardata.lst;

            for(iBase=1;iBase<=nbBases;iBase++)
            {
                bvect = B(, iBase);
                _simiComputeUvwCoord , stardata, bvect;
                if(!is_void(fixDelay))
                stardata.w = stardata.w - fixDelay(iBase);

                if(iBase==1)
                {
                    u = array(stardata.u, 1);
                    v = array(stardata.v, 1);
                    w = array(stardata.w, 1);
                    theta = array(stardata.theta, 1);
                    base = array(stardata.base, 1);
                }
                else
                {
                    grow, u, stardata.u;
                    grow, v, stardata.v;
                    grow, w, stardata.w;
                    grow, theta, stardata.theta;
                    grow, base, stardata.base;

                }
            }
            uvTable = [u, v];
            grow, uvwTable, [u, v, w];

            for(iBase=1;iBase<=nbBases;iBase++)
            {
                nPointsPerCircle = 30;
                param = span(-pi, pi, nPointsPerCircle);

                if(!strmatch(strpart(baseNames(iBase),1:2), "U"))
                {
                    diam = 1.8;
                }
                else if(!strmatch(strpart(baseNames(iBase),3:4), "U"))
                {
                    diam = 1.8;
                }
                else
                    diam = 8;

                u = uvTable(,1);
                v = uvTable(,2);

                if(colors=="bases")
                    Phases = array(5./6.*2*pi*(iBase-1)/double(nbBases-1),
                                   nLambda);
                else if(colors=="wlen")
                    Phases = span(1.5*pi,0,nLambda);
                else if(colors=="obs")
                    Phases = array(getPhase(kObs%7),nLambda);

                
                for(iP=1;iP<=nLambda;iP++)
                {
                    if(colors=="bases")
                        kolor = get_color(1,Phases(iP));
                    else if(colors=="wlen")
                        kolor = get_color(1,Phases(iP));
                    else if(colors=="obs")
                        kolor = getIntens(kObs%7)*get_color(1,Phases(iP));
                    else
                    {
                        kolor = getColVal(colors);
                        write,kolor,colors;
                    }
                    
                    grow,coloring,array(kolor,1);
                }
            }
        }

        tmax = spFreqFact*grow((v+diam/2)/min(lambda)/kilo2unit,
                               (u+diam/2)/min(lambda)/kilo2unit);
        tmin = spFreqFact*grow((v-diam/2)/min(lambda)/kilo2unit,
                               (u-diam/2)/min(lambda)/kilo2unit);
        //tmax = tmin = 1.6e5;
    }

    nha          = numberof(lst)/nObs;
    usedStations = reform(usedStations,[4,2,nbBases,nha,nObs]);
    uvwTable     = reform(uvwTable,[4,nbBases,3,nha,nObs]);
    uvwTable     = transpose(uvwTable,[1,2]);
    lst          = reform(lst,[2,nha,nObs]);
    coloring     = reform(coloring,[5,3,nLambda,nbBases,nha,nObs]);
    return uvwTable;
}

/************************************************************************/

func getColVal(color)
/* DOCUMENT getColVal(color)

   DESCRIPTION

   PARAMETERS
   - color: 

   RETURN VALUES

   CAUTIONS

   EXAMPLES

   SEE ALSO
*/
{
    if(color="red")
        return [255,0,0];
    else if(color="green")
        return [0,255,0];
    else if(color="blue")
        return [0,0,255];
    else if(color="cyan")
        return [0,255,255];
    else if(color="magenta")
        return [255,0,255];
    else if(color="yellow")
        return [255,255,0];
    else if(color="black")
        return [0,0,0];
    else if(color="white")
        return [255,255,255];
    else
        return int(255.*random(3));
}

/************************************************************************/ 

func simiPlotUV(stardata, stations, &lst, &coloring, &usedStations, hacen=, harange=, hastep=, lambda=, colors=, mx=, fp=, frac=, msize=, marker=, spFreqFact=, xyTitles=, place=, interferometer=)
/* DOCUMENT simiPlotUV(stardata, stations, &lst, &coloring, &usedStations, hacen=, harange=, hastep=, lambda=, colors=, mx=, fp=, frac=, msize=, marker=, spFreqFact=, xyTitles=, place=, interferometer=)

       DESCRIPTION

       PARAMETERS
   - stardata      : input star structure
   - stations      : input stations
   - lst           : (output) local sidereal time
   - coloring      : (output) colours for each base/observation/point
   - usedStations  : 
   - hacen         : hour angle centre
   - harange       : hour angle range
   - hastep        : hour angle step
   - lambda        : wavelength
   - colors        : colors
   - mx            : 
   - fp            : 
   - frac          : 
   - msize         : 
   - marker        : 
   - spFreqFact    : 
   - xyTitles      : 
   - place         : 
   - interferometer: 

       RETURN VALUES

       CAUTIONS

       EXAMPLES

       SEE ALSO
    */
{
        
    local uvTable, kObs, iObs, iBase, iP; 
    
    if(is_void(spFreqFact))
        spFreqFact = 1.0;
    
    if(is_void(lambda))
    {
        nLambda = 20;
        lambdaMin = 1.1 * micro2unit;
        lambdaMax = 2.5 * micro2unit;
        lambda = span(lambdaMax,lambdaMin,nLambda);
    }
    else
        nLambda = numberof(lambda);

    if(is_void(xyTitles))
        xyTitles = ["E <------ Spatial frequency (m^-1^)",
                    "Spatial frequency (m^-1^) ------> N"];

    if(is_void(msize))
        msize = 1.0;

    if(is_void(marker))
        marker = '\1';

    dimz = dimsof(stations);
    if(dimz(1)==1)
        nObs = 1;

    if(dimz(1)==2)
        nObs = dimz(0);

    if(is_void(mx))
        mx = 0;
    
    uvTable = simiGenerateUV(stardata, stations, lst, coloring, usedStations,hacen=hacen,
                             harange=harange, hastep=hastep, lambda=lambda,
                             colors=colors, mx=mx, fp=fp, frac=frac, msize=msize,
                             marker=marker, spFreqFact=spFreqFact,
                             xyTitles=xyTitles, place=place, interferometer=interferometer);

    nHa = dimsof(uvTable)(4);
    nLambda = numberof(lambda);

    for(kObs=1;kObs<=nObs;kObs++)
    {
        nbTel = numberof(stations(,kObs));
        nbBases = nbTel*(nbTel-1)/2;

        for(iObs=1;iObs<=nHa;iObs++)
        {
            for(iBase=1;iBase<=nbBases;iBase++)
            {
                u = uvTable(1,,iObs,kObs);
                v = uvTable(2,,iObs,kObs);

                for(iP=1;iP<=nLambda;iP++)
                {
                    kolor = coloring(,iP,iBase,iObs,kObs);

                    vspfreq = v(iBase)/lambda(iP);
                    uspfreq = u(iBase)/lambda(iP);

                    if(fp==1)
                    {
                        nPointsPerCircle = 30;
                        param = span(-pi, pi, nPointsPerCircle);

                        if(!strmatch(strpart(baseNames(iBase),1:2), "U"))
                        {
                            diam = 1.8;
                        }
                        else if(!strmatch(strpart(baseNames(iBase),3:4), "U"))
                        {
                            diam = 1.8;
                        }
                        else
                            diam = 8.0;

                        plfp,char(array(kolor,1)),
                            spFreqFact*(diam/2/lambda(iP)*cos(param)+vspfreq), 
                            spFreqFact*(diam/2/lambda(iP)*sin(param)+uspfreq),
                            nPointsPerCircle,edges=0;

                        plfp,char(array(kolor,1)),
                            spFreqFact*(diam/2/lambda(iP)*cos(param)-vspfreq), 
                            spFreqFact*(diam/2/lambda(iP)*sin(param)-uspfreq),
                            nPointsPerCircle,edges=0;
                    }
                    else
                    {
                        plg, spFreqFact*(vspfreq), spFreqFact*(uspfreq),
                            type="none",marker=marker,color=kolor,msize=msize;
                        plg, spFreqFact*(-vspfreq), spFreqFact*(-uspfreq),
                            type="none",marker=marker,color=kolor,msize=msize;

                    }
                }
            }
        }

        tmax = spFreqFact*grow((v+8.0/2)/min(lambda),
                               (u+8.0/2)/min(lambda));
        tmin = spFreqFact*grow((v-8.0/2)/max(lambda),
                               (u-8.0/2)/max(lambda));

        if(mx<max(abs(tmin)))
            mx = max(abs(tmin));
        if(mx<max(abs(tmax)))
            mx = max(abs(tmax));
        //tmax = tmin = 1.6e5;
    }

    limits, mx, -mx, -mx, mx;

    plg,0,0,type="none",marker='\2';

    xytitles, xyTitles(1), xyTitles(2), [0.00,0.02];

    return uvTable;
} 

/*****************************************************************************
 *
 * This part of code has been largely taken from the
 * Yorick plugin for AMBER Data Reduction Software
 *
 */

/***************************************************************************/

struct simiSTARVIS
{
    string date;
    double u;
    double v;
    double w;
    double ra;
    double dec;
    double lst;
    double ha;
    double theta;
    double base;
    double delay; 
};

/***************************************************************************/

struct simiLOCATION
{
    string name;
    double tz;
    double dst;
    double lon;
    double lat;
    double elev;
};

/***************************************************************************/

func simiGetLocation(place, interferometer, &loc)
/* DOCUMENT simiGetLocation(place, interferometer, &loc)

       DESCRIPTION
       This function fills in the input simiLOCATION structure fields with the
       location on .

       PARAMETERS
   - place         : 
   - interferometer: 
   - loc           : created simiLOCATION structure.

       EXAMPLES
       > _simiGetLocation(l)
       You are here : ESO, Cerro Paranal
       > l
       simiLOCATION(name="ESO, Cerro Paranal",tz=4,dst=-1,lon=-4.69365,
       lat=-24.6279,elev=2635)

       SEE ALSO
    */
{
    loc = simiLOCATION(0);
    if(place=="paranal")
    {
        if(is_void(interferometer))
            interferometer="VLTI";
        loc.name = "ESO, Cerro Paranal";
        loc.tz = 4;
        loc.dst = -1;
        loc.lon = -4.69365;
        loc.lat = -24.62794830;
        loc.elev = 2635;
        _simiSaveStations(interferometer);
    }
    else if(place=="e-elti")
    {
        loc.name = "ESO, Cerro Armazones";
        loc.tz = 4;
        loc.dst = -1;
        loc.lon = -4.69365;
        loc.lat = -24.62794830;
        loc.elev = 3003;
        _simiSaveStations(interferometer);
    }
    else if(place=="MWI")
    {
        if(is_void(interferometer))
            interferometer="CHARA";
        loc.name = "Mount Wilson";
        loc.tz   = 4;
        loc.dst  = -1;
        loc.lon  = -118.061644;
        loc.lat  =   34.223758;
        loc.elev = 1742;
        _simiSaveStations(interferometer);
    }
    else if(place=="calern")
    {
        if(is_void(interferometer))
            interferometer="GI2T";
    }
    else
        error,"Place not found!";
    /* Test output */
    yocoLogTest, swrite( format="You are here :\t%s\n", loc.name);
}

/***************************************************************************/

func _simiComputeUvwCoord(&data, bvect)
/* DOCUMENT _simiComputeUvwCoord(&data, bvect)

       DESCRIPTION
       Corrects uvw coordinates using base vector of observation bvect, and update
       data fields with results.

       PARAMETERS
   - data : simiSTARVIS structure to be corrected.
   - bvect: base vector of observation.

       SEE ALSO
    */
{
    data.ha = data.lst - data.ra;

    degr = 180 / pi;
    hour = degr / 15.;
    Bnorm = sqrt((bvect^2)(sum));

    /* Baseline vector in alt-az coordinates */
    Balt = asin(bvect(3) / (Bnorm+(Bnorm==0))) * degr;
    Baz = atan(bvect(2), bvect(1)) * degr;

    /* Baseline vector in equatorial coordinates */
    Bdec = asin(sin(Balt/degr) * sin(loc.lat/degr) + 
                cos(Balt/degr) * cos(loc.lat/degr) * cos(Baz/degr)) * degr;

    yBha = sin(Balt/degr) * cos(loc.lat/degr) - 
        cos(Balt/degr) * cos(Baz/degr) * sin(loc.lat/degr);
    zBha = -1. * cos(Balt/degr) * sin(Baz/degr);
    Bha = (atan(zBha,yBha) * hour + 24) % 24;

    /* baseline vector in the equatorial cartesian frame */
    Lx = - (-Bnorm * cos(Bdec/degr) * cos(Bha/hour));
    Ly = - Bnorm * cos(Bdec/degr) * sin(Bha/hour);
    Lz = Bnorm * sin(Bdec/degr);

    /* projection of the baseline vector on the u,v,w frame */
    data.u = (sin(data.ha/hour) * Lx + cos(data.ha/hour) * Ly);
    data.v = (- sin(data.dec/degr) * cos(data.ha/hour) * Lx 
              + sin(data.dec/degr) * sin(data.ha/hour) * Ly
              + cos(data.dec/degr) * Lz);
    data.w = (cos(data.dec/degr) * cos(data.ha/hour) * Lx 
              - cos(data.dec/degr) * sin(data.ha/hour) * Ly
              + sin(data.dec/degr) * Lz);

    data.theta = atan(data.u, data.v) * degr;
    data.base = sqrt(data.u^2 + data.v^2);
    data.delay = - data.w;
}

/***************************************************************************/

func simiGetNonRedundantStations(void, stationsFile=, nbTels=)
/* DOCUMENT simiGetNonRedundantStations(stationsFile=, nbTels=)

       DESCRIPTION

       PARAMETERS
   - stationsFile: 
   - nbTels      : 

       RETURN VALUES

       CAUTIONS

       EXAMPLES

       SEE ALSO
    */
{
    if(is_void(stationsFile))
        stationsFile="~/.interfStations.dat";

    dat = yocoFileReadAscii(stationsFile);
    nStat = numberof(dat(1,2:));
    stats = dat(1,2:);
    SE = yocoStr2Double(dat(4,2:));
    SN = yocoStr2Double(dat(5,2:));
    SZ = yocoStr2Double(dat(6,2:));
    diam = yocoStr2Double(dat(7,2:));
    count = 0;
    NBE = 1;
    BE = BN = BZ = BLEN = BANG = [0.0];
    st1 = st2 = st3 = [""];

    if(nbTels==2)
    {
        for(k=1;k<=nStat;k++)
        {
            for(l=k+1;l<=nStat;l++)
            {
                    be = SE(l)-SE(k);
                    bn = SN(l)-SN(k);
                    bz = SZ(l)-SN(k);
                    blen = abs(be,bn,bz);
                    bang = atan(be,bn);
                    
                    basdif = abs(BE-be,BN-bn);
                    
                    //write,bangdif
                    if(noneof(basdif < avg(diam)/2.)||
                       (stats(k)!="LAB")||
                       (stats(l)!="LAB"))
                    {
                        count++;
                        if(count>NBE)
                        {
                            grow,st1,array("",numberof(st1));
                            grow,st2,array("",numberof(st2));
                            grow,BE,array(99.0,numberof(BE));
                            grow,BN,array(99.0,numberof(BN));
                            grow,BZ,array(99.0,numberof(BZ));
                            grow,BLEN,array(99.0,numberof(BLEN));
                            grow,BANG,array(99.0,numberof(BANG));
                            NBE = numberof(BE);
                            write,count;
                        }
                        st1(count) = stats(k);
                        st2(count) = stats(l);
                        BE(count) = be;
                        BN(count) = bn;
                        BZ(count) = bz;
                        BLEN(count) = blen;
                        BANG(count) = bang;
                    }
            }
        }

    ST = transpose([st1(:count),st2(:count)]);
    }
    else if(nbTels==3)
    {
        for(k=1;k<=nStat;k++)
        {
            for(l=k+1;l<=nStat;l++)
            {
                for(m=l+1;m<=nStat;m++)
                {
                    if((stats(k)!="LAB")&&
                       (stats(l)!="LAB")&&
                       (stats(m)!="LAB"))
                    {
                        count++;
                        if(count>NBE)
                        {
                            grow,st1,array("",numberof(st1));
                            grow,st2,array("",numberof(st2));
                            grow,st3,array("",numberof(st3));
                            NBE = numberof(st1);
                            write,count;
                        }
                        st1(count) = stats(k);
                        st2(count) = stats(l);
                        st3(count) = stats(m);
                    }
                }
            }
        }

    ST = transpose([st1(:count),st2(:count),st3(:count)]);
    }
    else
        error,"Not yet coded!";
    return ST;
}

/***************************************************************************/ 

func _simiSaveStations(interferometer, stationsFile=)
/* DOCUMENT _simiSaveStations(interferometer, stationsFile=)

       DESCRIPTION
       Saves a table containing the stations coordinates in observatory
       coordinates (P & Q) and in E-W, N-S coordinates (metres).

       RESULTS
       the file ~/.positions.dat is written

       CAUTIONS
       This data comes from the ESO webpage
       http://www.eso.org/observing/etc/doc//baseline/tations.html
       which may not contain the most up-to-date positions of the stations.
       Therefore, an error can occur due to the obsolescence of this table.

       SEE ALSO
    */
{
    if(is_void(stationsFile))
        stationsFile="~/.interfStations.dat";

    // Paranal observatory, VLTI
    if(interferometer=="VLTI")
    {
        positions = " ID P Q E N ALT D C\n\
A0 -32.001 -48.013 -14.642 -55.812 0.01.8 1\n \
A1 -32.001 -64.021 -9.434 -70.949 0.0 1.8 2\n \
B0 -23.991 -48.019 -7.065 -53.212 0.0 1.8 3\n \
B1 -23.991 -64.011 -1.863 -68.334 0.0 1.8 4\n \
B2 -23.991 -72.011 0.739 -75.899 0.0 1.8 5\n  \
B3 -23.991 -80.029 3.348 -83.481 0.0 1.8 6\n  \
B4 -23.991 -88.013 5.945 -91.030 0.0 1.8 7\n  \
B5 -23.991 -96.012 8.547 -98.594 0.0 1.8 8\n  \
C0 -16.002 -48.013 0.487 -50.607 0.0 1.8 9\n \
C1 -16.002 -64.011 5.691 -65.735 0.0 1.8 10\n \
C2 -16.002 -72.019 8.296 -73.307 0.0 1.8 11\n \
C3 -16.002 -80.010 10.896 -80.864 0.0 1.8 12\n \
D0 0.010 -48.012 15.628 -45.397 0.0 1.8 13\n \
D1 0.010 -80.015 26.039 -75.660 0.0 1.8 14\n \
D2 0.010 -96.012 31.243 -90.787 0.0 1.8 15\n \
E0 16.011 -48.016 30.760 -40.196 0.0 1.8 16\n \
G0 32.017 -48.0172 45.896 -34.990 0.0 1.8 17\n \
G1 32.020 -112.010 66.716 -95.501 0.0 1.8 18\n \
G2 31.995 -24.003 38.063 -12.289 0.0 1.8 19\n \
H0 64.015 -48.007 76.150 -24.572 0.0 1.8 20\n \
I1 72.001 -87.997 96.711 -59.789 0.0 1.8 21\n \
J1 88.016 -71.992 106.648 -39.444 0.0 1.8 22\n \
J2 88.016 -96.005 114.460 -62.151 0.0 1.8 23\n \
J3 88.016 7.996 80.628 36.193 0.0 1.8 24\n \
J4 88.016 23.993 75.424 51.320 0.0 1.8 25\n \
J5 88.016 47.987 67.618 74.009 0.0 1.8 26\n \
J6 88.016 71.990 59.810 96.706 0.0 1.8 27\n \
K0 96.002 -48.006 106.397 -14.165 0.0 1.8 28\n \
L0 104.021 -47.998 113.977 -11.549 0.0 1.8 29\n \
M0 112.013 -48.000 121.535 -8.951 0.0 1.8 30\n \
U1 -16.000 -16.000 -9.925 -20.335 8.504 8 31\n \
U2 24.000 24.000 14.887 30.502 8.504 8 32\n \
U3 64.0013 47.9725 44.915 66.183 8.504 8 33\n \
U4 112.000 8.000 103.306 43.999 8.504 8 34\n \
LAB 52.000 -40.000 60 -20 0.0";
    }
    // Paranal observatory, VLT/NACO SAM 7 holes
    else if(interferometer=="NACO_SAM_7")
    {
        require,"yeti.i";   
        mask7HolesPos = transpose(Rotate_2D(transpose(
                                                      1.03*[[   3.51064,     -1.99373],
                                                            [   3.51064 ,     2.49014],
                                                            [    1.56907,      1.36918],
                                                            [   1.56907 ,     3.61111],
                                                            [  -0.372507,     -4.23566],
                                                            [   -2.31408,      3.61111],
                                                            [   -4.25565,     0.248215]]),3*deg2rad));
        mask7HolesDiam = 1.50;
        dmz = dimsof(mask7HolesPos);
        nm = dmz(4);
        positions = " ID P Q E N ALT D C\n";
        for(k=1;k<=nm;k++)
        {
            x = mask7HolesPos(1,k);
            y = mask7HolesPos(2,k);
            r = abs(x,y);
            
            stat = "H"+pr1(k)+
                " "+pr1(x) +
                " "+pr1(y) +
                " "+pr1(x) +
                " "+pr1(y) +
                " 0.0" +
                " "+pr1(mask7HolesDiam)+
                " "+pr1(1);
            positions = positions + stat + "\n";
        }
        positions = positions + "LAB 0.0 0.0 0.0 0.0 0.0 0.0 1";
    }
    // Mount, Wilson, CHARA array
    else if(interferometer=="CHARA")
    {
        positions = " ID P Q E N ALT D C\n\
S1  0.0 0.0 0.0 0.0 0.0 1.0 1\n   \
S2  0.0 0.0   -5.747952926  33.576627      0.637472388 1.0 2\n   \
E1  0.0 0.0  125.333133300 305.928497300  -5.919099700 1.0 3\n   \
E2  0.0 0.0   70.3891451   269.714687100  -2.802564400 1.0 4\n   \
W1  0.0 0.0 -175.068410100 216.327246400 -10.797526100 1.0 5\n   \
W2  0.0 0.0  -69.084592500 199.342434600   0.470608600 1.0 6\n   \
LAB 0.0 0.0  -20.0         200 0.0 0.0 7";
        // earth radius 6 371,0
        //W2 34.22634 -118.05892
        //W1 34.22617 -118.05777
        //E1 34.226794 -118.05625
        //E2 34.227128 -118.055655
        //S1 34.22469 -118.057084
        //S2 34.224374 -118.057020
        //obsevatory loc.lon  = -118.061644;
        //observatory loc.lat  =   34.223758;
        // rEarth = 6371.0;
        //     x = ([34.22634,
        //           34.22617,
        //           34.226794,
        //           34.227128,
        //           34.22469 ,
        //           34.224374]-34.223758)*pi/180*rEarth*1000;
        //     y = ([-118.05892,
        //           -118.05777,
        //           -118.05625,
        //           -118.055655,
        //           -118.057084,
        //           -118.057020]+118.061644)*pi/180*rEarth*1000;
    }
    else if(interferometer=="susi")
    {
        //-30.31780 149.54935
        //-30.31874 149.54931
        //-30.32022 149.54926
        //-30.32093 149.54933
        //-30.32165 149.54929
        //-30.32289 149.54926
        //-30.32361 149.54926
        //-30.32071 149.54929
    }
    else if(interferometer=="calern")
    {
        positions = " ID P Q E N ALT D C\n\
M1  0.0 0.0  0.0  0.0   0.0 1.5 1\n\
G1  0.0 0.0 -151 -157   0.0 1.5 2\n\
G2  0.0 0.0 -151 -141.5 0.0 1.5 3\n\
G1b 0.0 0.0 -151 -181.5 0.0 1.5 4\n\
G2b 0.0 0.0 -151 -109.5 0.0 1.5 5\n\
C1  0.0 0.0  146 -103   0.0 1.0 6\n\
C2  0.0 0.0  167 -103   0.0 1.0 7\n\
S1  0.0 0.0 -12  -401.5 0.0 1.5 8";

        // MeO     43.754655, 6.921705
        // C2PU2   43.753716, 6.923037
        // C2PU2   43.753716, 6.923229
        // Schmidt 43.750994, 6.921594
        // GI2T1   43.753225, 6.920325
        // GI2T2   43.753366, 6.920317
        // GI2T1b  43.753000, 6.920329
        // GI2T2b  43.753657, 6.920332
    }
    else if(interferometer=="pti")
    {
        //33.357159 -116.864469
        //33.357424 -116.863590
        //33.356481 -116.863995
    }
    // Paranal observatory, E-ELTI ;-)
    else if(interferometer=="e-elti")
    {
        positions = " ID P Q E N ALT D C\n";
        stations = [];
        // ELT diameter
        D = 42
            //   D = 39.9
        // ELT central obscuration
        //D2 = 0;
        D2 = 11.76;
         D2 = 11.2
             //    D2 = 23;
        //d = 10;
        d = 1.47*cos(pi/6);
        longBase = 300;
        par = span(0,2*pi,1000);

        nm = int(D/d*sqrt(2))+1;

        count=0;
        clust = 1;
        for(k=-nm/2;k<=nm/2;k++)
        {
            for(l=-nm/2;l<=nm/2;l++)
            {
                y = k*d + 0.5*d*(l%2)+0.5*d;
                x = (l-1)*d*sin(pi/3);
                r = abs(x,y);
                if((r>D2/2+d/2)&&(r<D/2-d/2))
                {
                    count ++;
                    stat = "O"+pr1(count)+
                        " "+pr1(x) +
                        " "+pr1(y) +
                        " "+pr1(x) +
                        " "+pr1(y) +
                        " 0.0" +
                        " "+pr1(d)+
                        " "+pr1(clust);
                    positions = positions + stat + "\n";
                    grow,stations,"O"+pr1(count);
                }
            }
        }

        count=0;
        maxr = 600;
        for(ang=-2*pi/3;ang<=6*pi/3;ang+=4*pi/3)
        {
            write,ang;
            for(k=D+D/2-10;k<=longBase;k+=40)
            {
                clust++;
                r = double(k);
                rounds = 1;
                x = r*cos(ang);
                y = r*sin(ang);
                r = abs(x,y);
                
                count ++;
                stat = "A"+pr1(count)+
                    " "+pr1(x) +
                    " "+pr1(y) +
                    " "+pr1(x) +
                    " "+pr1(y) +
                    " 0.0" +
                    " "+pr1(d)+
                    " "+pr1(clust);
                positions = positions + stat + "\n";
                grow,stations,"A"+pr1(count);
                
            }
        }
    }
    else
        error,"Interferometer not found!"

            /* Save the station */
            fh = open(stationsFile, "w");
    write, fh, positions;
    close, fh;

    /* Verbose output */
    yocoLogInfo,"Station saved into file",stationsFile;
    return stations;
}

/***************************************************************************/ 

func simiGetBaseVect(station1, station2, &orig, &fixDelay, stationsFile=)
/* DOCUMENT simiGetBaseVect(station1, station2, &orig, &fixDelay, stationsFile=)

       DESCRIPTION
       Gets the geographical coordinates based on stations.

       PARAMETERS
   - station1    : first station for considered baseline
   - station2    : second station for considered baseline
   - orig        : geagraphical origin coordinates.
   - fixDelay    : 
   - stationsFile: 

       SEE ALSO
    */
{
    if(is_void(stationsFile))
        stationsFile = "~/.interfStations.dat";

    if(!open(stationsFile, "r",1))
        error,"Please use simiGetLocation first";

    //xy legends ajusting
    adjust = [0.025,0.025];

    dat = yocoFileReadAscii(stationsFile);
    nStat = numberof(dat(1,));
    for(k=2;k<=nStat;k++)
    { 
        if (dat(1,k) == station1)
        {
            BP1 = yocoStr2Double(dat(2,k));
            BQ1 = yocoStr2Double(dat(3,k));
            BE1 = yocoStr2Double(dat(4,k));
            BN1 = yocoStr2Double(dat(5,k));
            BZ1 = yocoStr2Double(dat(6,k));
            
            orig = [BE1, BN1, BZ1];
        }
        else if (dat(1,k) == station2)
        {
            BP2 = yocoStr2Double(dat(2,k));
            BQ2 = yocoStr2Double(dat(3,k));
            BE2 = yocoStr2Double(dat(4,k));
            BN2 = yocoStr2Double(dat(5,k));
            BZ2 = yocoStr2Double(dat(6,k));
        }
        else if (dat(1,k) == "LAB")
        {
            LABP = yocoStr2Double(dat(2,k));
            LABQ = yocoStr2Double(dat(3,k));
            LABE = yocoStr2Double(dat(4,k));
            LABN = yocoStr2Double(dat(5,k));
            LABZ = yocoStr2Double(dat(6,k));
            lab = [LABE, LABN, LABZ];
            DL1 = sum([abs(BQ1-LABQ), abs(BP1-LABP)]);
            DL2 = sum([abs(BQ2-LABQ), abs(BP2-LABP)]);
            fixDelay = DL2-DL1;
            write,station1, DL1, station2, DL2, "Fixed delay",fixDelay;
        }
    }

    /* Test output */
    yocoLogTest,
        swrite(format = station1 + "-" + station2 +
               " // base =%fm, angle=%fdeg",
               sqrt((BE2-BE1)^2 + (BN2-BN1)^2),
               180 / pi * atan(BN2-BN1, BE2-BE1) );

    return [BN2-BN1, BE2-BE1, BZ2-BZ1];
}

/************************************************************************/ 

func simiPlotStations(color=, stationsFile=, noLabels=, plotN=)
/* DOCUMENT simiPlotStations(color=, stationsFile=, noLabels=, plotN=)

       DESCRIPTION
       Plots different stations.

       PARAMETERS 
   - color       : color function for plot.
   - stationsFile: 
   - noLabels    : 
   - plotN       : 

       SEE ALSO
    */
{
    if(is_void(stationsFile))
        stationsFile = "~/.interfStations.dat";

    if(!open(stationsFile, "r",1))
        error,"Please use simiGetLocation first";

    fh = open(stationsFile, "r");
    pos = "";
    BP = BQ = BE = BN = 0.0;
    read, fh, format="%s %f %f %f %f", pos, BP, BQ, BE, BN;

    if(is_void(plotN))
        plotN = 100;
    
    param = span(-pi+pi/6, pi+pi/6, plotN+1);

    while (read(fh, format="%s %f %f %f %f", pos, BP, BQ, BE, BN))
    {
        if (strmatch(pos, "U"))
        {
            diam = 8;
            diam2 = 26;
            just = "CH";
        }
        else
        {
            diam = 1.8;
            diam2 = [];
            just = "RT";
        }

        if(!is_void(diam2))
            plg, diam2/2.*sin(param) + BN, diam2/2.*cos(param) + BE, color=[128,128,128],marks=0;

        if (strmatch(pos, "LAB"))
        {
            plmk,BN, BE, color=color, marker=2, msize=0.2;
        }
        else
            plg, diam/2.*sin(param) + BN, diam/2.*cos(param) + BE, color=color,marks=0;
        if(noLabels!=0)
            plt, pos, BE, BN, tosys=1, height=6, justify=just, color=color;
    }
    
    if(noLabels!=0)
        xytitles, "position (m) -----> E",
            "position (m) -----> N",
            [0.00,0.02];
    
    limits, -53, 157, -105, 105;
}

/************************************************************************/ 

func simiComputeBaseVect(&orig, &stations, &baseNames, &fixDelay, &usedStations, fileName=)
/* DOCUMENT simiComputeBaseVect(&orig, &stations, &baseNames, &fixDelay, &usedStations, fileName=)

       DESCRIPTION
       Computes ground baselines of the from a given ESO raw data file,
       or from a list of stations names.

       PARAMETERS
   - orig        : geagraphical telescopes origin coordinates.
   - stations    : telescopes stations for considered baseline
   - baseNames   : returned base names using the stations names
   - fixDelay    : 
   - usedStations: 
   - fileName    : name of file where stations information are extracted.

       RETURN VALUES
       ground baselines vectors. 

       SEE ALSO
    */
{
    keys = ["HIERARCH ESO ISS CONF STATION1",
            "HIERARCH ESO ISS CONF STATION2",
            "HIERARCH ESO ISS CONF STATION3"];

    if (is_void(stations))
    {
        message = "FILE NAME\nEnter a file";
        if (amdlibFileChooser(message,fileName) == 0)
        {
            return 0;
        }

        valuesTab = _simiGetKwdVals(fileName, keys,gui=0);

        stations = [valuesTab(1),valuesTab(2),valuesTab(3)];
        stations = stations(where(stations!="-"));
    }
    if (is_void(stations))
    {
        yocoGuiInfoBox, "failing to get AMBER stations information";
        return 0;
    }
    nbTels = numberof(stations);
    nbBases = nbTels * (nbTels - 1) / 2;

    B = baseNames = orig = fixDelay = [];

    st1 = st2 = [0];
    count = 0;
    for(i=1;i<=nbTels;i++)
    {
        for(j=1;j<=i-1;j++)
        {
            count++;
            if(count>numberof(st1))
            {
                grow, st1, array(0,numberof(st1));
                grow, st2, array(0,numberof(st2));
                write,count;
            }
            st1(count)=j;
            st2(count)=i;
        }
    }

    //Determination du vecteur de base de l'observation
    for (iBase=1; iBase <= nbBases; iBase++)
    {
        grow, B, array(simiGetBaseVect(stations(st1(iBase)),
                                       stations(st2(iBase)), origP, fixDL), 1);
        if(!is_void(fixDL))
            grow,fixDelay, array(fixDL, 1);
        grow,baseNames,stations(st1(iBase)) + stations(st2(iBase))
            grow, orig, array(origP, 1);
        grow,usedStations,[[stations(st1(iBase)), stations(st2(iBase))]];
    }

    return B;
}

/************************************************************************/

func simiComputUvCoord(&orig, &B, &u, &v, &w, &stations, file=)
/* DOCUMENT simiComputUvCoord(&orig, &B, &u, &v, &w, &stations, file=)

       DESCRIPTION
       This function computes UV coordinates with an AMBER fits
       file given as input. You can then compare them with the UV
       coordinates of the file header

       PARAMETERS
   - orig    : geagraphical telescop origin coordinates.
   - B       : projected baselines vector.
   - u       : Fourier space coordinates.
   - v       : Fourier space coordinates.
   - w       : Fourier space coordinates.
   - stations: stations
   - file    : input file.

       RETURN VALUES
       1 on succesful completion, 0 otherwise

       SEE ALSO
       simiComputeBaseVect, _simiComputeUvwCoord
    */
{
    stardata = simiSTARVIS(0);

    // Select cerro paranal
    _simiGetLocation, interferometer, loc;

    keys = ["DATE-OBS", "RA", "DEC", "LST"];

    message = "FILE NAME\nEnter a file";
    if (amdlibFileChooser(message,file) == 0)
    {
        return 0;
    }

    valuesTab = _simiGetKwdVals(file, keys,gui=0);

    if (valuesTab(1) == "-")
    {
        yocoGuiInfoBox, "Not enough information in the file header";
        return 0;
    }

    dateObs = valuesTab(1);
    dateObsStr = yocoStrSplit(dateObs, "T");
    dateStr = dateObsStr(1);
    timsStr = dateObsStr(2);
    time = yocoStr2Double(yocoStrSplit(timsStr, ":"));

    date = yocoStrSplit(dateStr, "-");
    day = date(3);
    month = date(2);
    year = date(1);
    stardata.date = day + "/" + month + "/" + year;

    stardata.lst = yocoStr2Double(valuesTab(4)) / 3600;

    // Detemine observed star's coordinates
    RA = yocoStr2Double(valuesTab(2));
    stardata.ra = RA/15.0;

    DEC = yocoStr2Double(valuesTab(3));
    stardata.dec = DEC;

    /* Test output */
    yocoLogTest, "You observed at " + timsStr + " " + stardata.date;

    // Determine observation's base vector
    B = simiComputeBaseVect(fileName=file, orig, stations);

    if (is_void(B))
    {
        /* In this case, information necessary for computing UV coordinates are
         * not present in primary header. So nothing is written, and no graphs
         * relative to uv coordinates have to be plot. */
        return 0;
    }
    nbBases = numberof(B(1, ));
    for (iBase=1; iBase <= nbBases; iBase++)
    {
        bvect = B(,iBase);

        //Calcul du vecteur de base dans l'espace reciproque
        _simiComputeUvwCoord, stardata, bvect;

        if (iBase == 1)
        {
            u = array(stardata.u, 1);
            v = array(stardata.v, 1);
            w = array(stardata.w, 1);
            theta = array(stardata.theta, 1);
            base = array(stardata.base, 1);
        }
        else
        {
            grow, u, stardata.u;
            grow, v, stardata.v;
            grow, w, stardata.w;
            grow, theta, stardata.theta;
            grow, base, stardata.base;

        }
    }

    /* Test output */
    yocoLogTest, swrite(format="u: %f v: %f w: %f (m)", u,v,w);
    yocoLogTest, swrite(format="projected base length: %fm, angle: deg", base, theta);

    // write, "u :", u, "m, v :", v, "m, w :", w, "m";
    // write, "projected baseline :", base,
    // "m, projected angle :", theta, "°";

    return 1; 
}

/***************************************************************************/

func simiGetUVCoordinates(&w, fileName=)
/* DOCUMENT simiGetUVCoordinates(&w, fileName=)

       DESCRIPTION
       Compute UV coordinates with an AMBER fits file given as input.

       PARAMETERS
   - w       : 
   - fileName: name of file where uv coordinates have to be extracted.

       RETURN VALUES
       The uv coordinates array.

       SEE ALSO
    */
{
    nFiles = numberof(fileName);

    uvTable = [];

    for (iFile=1; iFile <= numberof(fileName); iFile++)
    {
        if (!simiComputUvCoord(orig, B, u, v, w, stations, 
                               file=fileName(iFile)))
        {
            yocoGuiInfoBox, "failing to process AMBER UV information";
            return 0;
        }
        grow, uvTable, [u,v,w];
    }

    return transpose(uvTable, [1,2]);
}

/***************************************************************************/

func simiPlotUVCoordinates(&uvTable, fileName=, figDir=, nameString=, noLabels=, noTitle=, baseNames=, color=, stations=, width=)
/* DOCUMENT simiPlotUVCoordinates(&uvTable, fileName=, figDir=, nameString=, noLabels=, noTitle=, baseNames=, color=, stations=, width=)

       DESCRIPTION
       This function plots UV coordinates.

       PARAMETERS
   - uvTable   : array containing corrected uv coordinates for each frame.
   - fileName  : name of observation file. Useful if no uvTable has 
       already been computed. 
   - figDir    : name of the directory where results are saved as ps 
       files. If not specified, graphs are not saved.
   - nameString: name of the bserved star.
   - noLabels  : if set to 0, labels are not plotted.
   - noTitle   : if set to 0, title are not plotted.
   - baseNames : names for each baseline
   - color     : color of the plot
   - stations  : input stations if not present in the file name
   - width     : 

       RETURN VALUES
       Returns 1 on sucessful completion, 0 otherwise.

       SEE ALSO
       plg
    */
{
    colors=["red","green","blue","cyan","magenta","yellow","black"];

    if (is_void(noTitle))
    {
        noTitle = 0;
    }
    if (is_void(noLabels))
    {
        noLabels = 0;
    }

    if (is_void(uvTable))
    {
        message = "FILE NAME\nEnter a file";
        if (amdlibFileChooser(message, fileName) == 0)
        {
            return 0;
        }
    }

    if (is_void(nameString))
    {
        titleString1 = " BASELINES";
        titleString2 = " PROJECTED BASELINES";
    }
    else
    {
        titleString1 = " BASELINES\n" + nameString;
        titleString2 = " PROJECTED BASELINES\n" + nameString;
    }

    nbFiles = numberof(fileName);

    if (nbFiles == 0)
    {
        if (is_void(uvTable))
        {
            yocoGuiInfoBox, "No files to get UV coordinates";
            return 0;
        }
        else
            nbFiles = dimsof(uvTable)(4);
    }

    if (is_void(uvTable))
    {
        for (iFile=1; iFile <= nbFiles; iFile++)
        {
            orig = B = u = v = w = stations = [];
            if (!simiComputUvCoord(orig, B, u, v, w, stations,
                                   file=fileName(iFile)))
            {
                yocoGuiInfoBox, "failing to process AMBER UV information";
                return 0;
            }

            grow, uvTable, array(transpose([u,v]),1);
        }

        baseNames = [];
        nbBases = dimsof(uvTable)(3);
        if(nbBases==1)
            nbTel = 2;
        else if(nbBases==3)
            nbTel = 3;
        tels = indgen(nbTel);
        for (iBase = 1; iBase <= nbBases; iBase++)
        {
            telUsed = int(roll(tels,1-iBase));
            if(is_void(stations))
                grow, baseNames, "";
            else
                grow, baseNames, pr1(iBase)+" = "+
                    stations(telUsed(1))+"-"+stations(telUsed(2));
        }

    }

    nPointsPerCircle = 30;
    param = span(-pi, pi, nPointsPerCircle);
    nbBases = dimsof(uvTable)(3);

    // (U,V) plot itself
    for (iFile=1; iFile <= nbFiles; iFile++)
    {
        for (iBase=1; iBase <= nbBases; iBase++)
        {
            telUsed = int(roll(tels,1-iBase));
            if (!strmatch(stations(telUsed(1)), "U"))
            {
                diam = 1.8;
            }
            else
            {
                diam = 8;
            }
            u = uvTable(1, ..);
            v = uvTable(2, ..);

            kolor = colors(iBase%7);
            if(!is_void(color))
                kolor = color;

            plg, diam / 2 * sin(param) + v(iBase,iFile), 
                diam / 2 * cos(param) + u(iBase,iFile), 
                color=kolor, width=width;
            plg, diam / 2 * sin(param) - v(iBase,iFile), 
                diam / 2 * cos(param) - u(iBase,iFile),
                color=kolor, width=width;
            plg, [0,0], [0,0], marks=1, marker='\2',type="none";

            tmax = grow(v + diam / 2, u + diam / 2);
            tmin = grow(v - diam / 2, u - diam / 2);

            mx = max(max(tmax), -min(tmin))*1.1;
        }
    }

    limits, mx, -mx, -mx, mx;

    // Labellize the graph
    if(noLabels==0)
    {
        for (iBase=1; iBase <= nbBases; iBase++)
        {
            Base = sqrt(u(iBase,avg)^2 + v(iBase,avg)^2);
            Theta = 180/pi*atan(u(iBase,avg),v(iBase,avg));

            if(!is_void(baseNames))
                text1 = " "+baseNames(iBase);
            else
                text1 = " "+pr1(iBase);
            plt, text1+ " : " + 
                strpart(pr1(Base), 1:5) + 
                "m, "+ strpart(pr1(Theta),1:5) + "°",
                mx, mx - mx/9 * iBase, 
                tosys=1, color=colors(iBase%7);

            xytitles, "E <----- U (m)", "V (m) -----> N",[0.00,0.02];
        }
    }
    if(noTitle==0)
    {
        pltitle, titleString2;
    }

    // Hard copy
    if (!is_void(figDir))
    { 
        fileHeader = getGoodName(nameString);
        hcps, figDir + "/" + fileHeader + "_proj_baselines.ps";
        yocoLogInfo, "Saved the 2 graphs";
    }

    return 1; 
}

/***************************************************************************/

func simiPlotUsedStations(&uvTable, &stations, fileName=, figDir=, color=, off=, type=, noTitle=, noLabels=)
/* DOCUMENT simiPlotUsedStations(&uvTable, &stations, fileName=, figDir=, color=, off=, type=, noTitle=, noLabels=)

       DESCRIPTION
       This function plots the used stations corresponding to
       an observation.

       PARAMETERS
   - uvTable : array containing corrected uv coordinates for each frame.
   - stations: 
   - fileName: name of observation file. Useful if no uvTable has 
       already been computed. 
   - figDir  : name of the directory where results are saved as ps
       files. If not specified, graphs are not saved.
   - color   : optional color of the plot
   - off     : optional offset (in meters)
   - type    : optional line type
   - noTitle : 
   - noLabels: 

       RETURN VALUES
       Returns 1 on sucessful completion, 0 otherwise.

       SEE ALSO
    */
{
    colors=["red","green","blue","cyan","magenta","yellow","black"];

    if (is_void(noTitle))
    {
        noTitle =0;
    }
    if (is_void(noLabels))
    {
        noLabels = 0;
    }
    if (is_void(off))
    {
        // offset of plot
        off = 0;
    }
    if (is_void(uvTable))
    {
        message = "FILE NAME\nEnter a file";
        if (amdlibFileChooser(message, fileName) == 0)
        {
            return 0;
        }
    }

    nbFiles = numberof(fileName);

    if (nbFiles == 0)
    {
        if (is_void(uvTable))
        {
            yocoGuiInfoBox, "No files to get UV coordinates";
            return 0;
        }
    }

    simiPlotStations, color="black";

    if (is_void(stations))
    {
        B = simiComputeBaseVect(orig, stations,
                                fileName=fileName(1));
    }
    else
    {
        B = simiComputeBaseVect(orig, stations);
    }
    nbTels = numberof(stations);
    nbBases = nbTels*(nbTels-1)/2;
    for (iBase=1; iBase <= nbBases; iBase++)
    {
        if(!is_void(color))
            kolor = color;
        else
        {
            kolor = colors(iBase%7);
            Phases = array(5./6.*2*pi*(iBase-1)/double(nbBases-1),
                           nLambda);
            kolor = get_color(1,Phases(iP));
        }
        
        pldj, orig(1, iBase)+off*sin(-20*deg2rad),
            orig(2, iBase)+off*cos(-20*deg2rad),
            orig(1, iBase) + B(2, iBase)+off*sin(-20*deg2rad), 
            orig(2, iBase) + B(1, iBase)+off*cos(-20*deg2rad), 
            color=kolor, type=type;
    }

    // Labelling the graph
    if (noLabels == 0)
    {
        // for (iBase=1; iBase <= nbBases; iBase++)
        // {
        //     Phases = array(5./6.*2*pi*(iBase-1)/double(nbBases-1),
        //                    nLambda);
        //     kolor = get_color(1,Phases(iP));
        
        //     if(!is_void(color))
        //         kolor = color;
        
        //     // plt, "Base " + pr1(iBase), 
        //     //     0.185, 0.415 + 0.0155 * (nbBases - iBase+1), 
        //     //     tosys=0, color=kolor;
        
        //     plt, "Base " + pr1(iBase), 
        //         -45, 100 - 90 * (iBase - 1) / double(nbBases-1), 
        //         tosys=plsys(), color=kolor, justify="LT";
        // }
    }
    if (NO_CRITE == 0)
    {
        simiCopyright;
    }

    // Hard copy
    if (!is_void(figDir))
    { 
        fileHeader = getGoodName(nameString);
        hcps, figDir + "/" + fileHeader + "__baselines.ps";
        yocoLogInfo, "Saved the 2 graphs";
    }

    return 1; 
}

/***************************************************************************/
