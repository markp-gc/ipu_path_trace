/**
  This library contains code for transcendental math. It completely avoids any use of doubles with
  the consequential limits on precision. Its purpose is speed not precision. If you need better
  precision use another libary. For example the C++ std library (which falls back to slow double
  emulation on machines without hardware double support).

  All the code in this library is derived from https://netlib.org/cephes under the license
  terms given in this file: https://netlib.org/cephes/readme reproduced below:

    Some software in this archive may be from the book _Methods and
    Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
    International, 1989) or from the Cephes Mathematical Library, a
    commercial product. In either event, it is copyrighted by the author.
    What you see here may be used freely but it comes with no support or
    guarantee.

    The two known misprints in the book are repaired here in the
    source listings for the gamma function and the incomplete beta
    integral.

    Stephen L. Moshier
    moshier@na-net.ornl.gov
**/

#pragma once

/*							tanf.c
 *
 *	Circular tangent
 *
 *
 *
 * SYNOPSIS:
 *
 * float x, y, tanf();
 *
 * y = tanf( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the circular tangent of the radian argument x.
 *
 * Range reduction is modulo pi/4.  A polynomial approximation
 * is employed in the basic interval [0, pi/4].
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     +-4096        100000     3.3e-7      4.5e-8
 *
 * ERROR MESSAGES:
 *
 *   message         condition          value returned
 * tanf total loss   x > 2^24              0.0
 *
 */

/*							cotf.c
 *
 *	Circular cotangent
 *
 *
 *
 * SYNOPSIS:
 *
 * float x, y, cotf();
 *
 * y = cotf( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the circular cotangent of the radian argument x.
 * A common routine computes either the tangent or cotangent.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE     +-4096        100000     3.0e-7      4.5e-8
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition          value returned
 * cot total loss   x > 2^24                0.0
 * cot singularity  x = 0                  MAXNUMF
 *
 */

/*
Cephes Math Library Release 2.2:  June, 1992
Copyright 1984, 1987, 1989 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

/* Single precision circular tangent
 * test interval: [-pi/4, +pi/4]
 * trials: 10000
 * peak relative error: 8.7e-8
 * rms relative error: 2.8e-8
 */

constexpr float DP1 = 0.78515625f;
constexpr float DP2 = 2.4187564849853515625e-4f;
constexpr float DP3 = 3.77489497744594108e-8f;
constexpr float FOPI = 1.27323954473516f;  /* 4/pi */
constexpr float lossth = 8192.f;

constexpr float MAXNUMF = 3.4028234663852885981170418348451692544e38f;
constexpr float MAXLOGF = 88.72283905206835f;
constexpr float MINLOGF = -103.278929903431851103f; /* log(2^-149) */

constexpr float LOG2EF = 1.44269504088896341f;
constexpr float LOGE2F = 0.693147180559945309f;
constexpr float SQRTHF = 0.707106781186547524f;
constexpr float PIF = 3.141592653589793238f;
constexpr float PIO2F = 1.5707963267948966192f;
constexpr float PIO4F = 0.7853981633974483096f;
constexpr float MACHEPF = 5.9604644775390625E-8f;

float tancotf( float xx, int cotflg )
{
float x, y, z, zz;
long j;
int sign;

/* make argument positive but save the sign */
if( xx < 0.f )
	{
	x = -xx;
	sign = -1;
	}
else
	{
	x = xx;
	sign = 1;
	}

if( x > lossth )
	{
	return(0.f);
	}

/* compute x mod PIO4 */
j = FOPI * x; /* integer part of x/(PI/4) */
y = j;

/* map zeros and singularities to origin */
if( j & 1 )
	{
	j += 1;
	y += 1.f;
	}

z = ((x - y * DP1) - y * DP2) - y * DP3;

zz = z * z;

if( x > 1.0e-4f )
	{
/* 1.7e-8 relative error in [-pi/4, +pi/4] */
	y =
	((((( 9.38540185543E-3f * zz
	+ 3.11992232697E-3f) * zz
	+ 2.44301354525E-2f) * zz
	+ 5.34112807005E-2f) * zz
	+ 1.33387994085E-1f) * zz
	+ 3.33331568548E-1f) * zz * z
	+ z;
	}
else
	{
	y = z;
	}

if( j & 2 )
	{
	if( cotflg )
		y = -y;
	else
		y = -1.f/y;
	}
else
	{
	if( cotflg )
		y = 1.f/y;
	}

if( sign < 0 )
	y = -y;

return( y );
}

float tanf( float x )
{
return( tancotf(x,0) );
}



/*							atanf.c
 *
 *	Inverse circular tangent
 *      (arctangent)
 *
 *
 *
 * SYNOPSIS:
 *
 * float x, y, atanf();
 *
 * y = atanf( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns radian angle between -pi/2 and +pi/2 whose tangent
 * is x.
 *
 * Range reduction is from four intervals into the interval
 * from zero to  tan( pi/8 ).  A polynomial approximates
 * the function in this basic interval.
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      -10, 10     100000      1.9e-7      4.1e-8
 *
 */
/*							atan2f()
 *
 *	Quadrant correct inverse circular tangent
 *
 *
 *
 * SYNOPSIS:
 *
 * float x, y, z, atan2f();
 *
 * z = atan2f( y, x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns radian angle whose tangent is y/x.
 * Define compile time symbol ANSIC = 1 for ANSI standard,
 * range -PI < z <= +PI, args (y,x); else ANSIC = 0 for range
 * 0 to 2PI, args (x,y).
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      -10, 10     100000      1.9e-7      4.1e-8
 * See atan.c.
 *
 */

/*							atan.c */


/*
Cephes Math Library Release 2.2:  June, 1992
Copyright 1984, 1987, 1989, 1992 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

/* Single precision circular arcsine
 * test interval: [-tan(pi/8), +tan(pi/8)]
 * trials: 10000
 * peak relative error: 7.7e-8
 * rms relative error: 2.9e-8
 */

float atanf( float xx )
{
float x, y, z;
int sign;

x = xx;

/* make argument positive and save the sign */
if( xx < 0.f )
	{
	sign = -1;
	x = -xx;
	}
else
	{
	sign = 1;
	x = xx;
	}
/* range reduction */
if( x > 2.414213562373095f )  /* tan 3pi/8 */
	{
	y = PIO2F;
	x = -( 1.f/x );
	}

else if( x > 0.4142135623730950f ) /* tan pi/8 */
	{
	y = PIO4F;
	x = (x-1.f)/(x+1.f);
	}
else
	y = 0.f;

z = x * x;
y +=
((( 8.05374449538e-2f * z
  - 1.38776856032E-1f) * z
  + 1.99777106478E-1f) * z
  - 3.33329491539E-1f) * z * x
  + x;

if( sign < 0 )
	y = -y;

return( y );
}


float atan2f( float y, float x )
{
float z, w;
int code;


code = 0;

if( x < 0.f )
	code = 2;
if( y < 0.f )
	code |= 1;

if( x == 0.f )
	{
	if( code & 1 )
		{
		return( -PIO2F );
		}
	if( y == 0.f )
		return( 0.f );
	return( PIO2F );
	}

if( y == 0.f )
	{
	if( code & 2 )
		return( PIF );
	return( 0.f );
	}


switch( code )
	{
	default:
	case 0:
	case 1: w = 0.f; break;
	case 2: w = PIF; break;
	case 3: w = -PIF; break;
	}

z = atanf( y/x );

return( w + z );
}


