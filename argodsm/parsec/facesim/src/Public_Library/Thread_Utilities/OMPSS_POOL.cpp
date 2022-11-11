/*
	Copyright 2014-2015 Raul Vidal Ortiz - Barcelona Supercomputing Center
	Distributed under the same license as the rest of PhysBAM
*/
#include "OMPSS_POOL.h"


using namespace PhysBAM;

OMPSS_POOL* OMPSS_POOL::singleton_instance = 0;

OMPSS_POOL::
OMPSS_POOL ()
{
    this->number_of_divisions = 0;
}

OMPSS_POOL::
~OMPSS_POOL () {}

void OMPSS_POOL::
printBSClogo()
{
    std::cerr << 
    "          OOO                                          " << std::endl <<
    "      OOO      OO                                      " << std::endl <<
    "    OOO   OOOO    OOOOOO                               " << std::endl <<
    "  OOO   OOO   OOOOOOOOOOOOOO         BARCELONA         " << std::endl <<
    " OOO   OOO  OOOOOOOOOOOOOOOOOO                         " << std::endl <<
    "OOO   OOO  OOOOOOOOOOOOOOOOOOOO                        " << std::endl <<
    "OOO  OOO  OOOOOOOOOOOOOOOOOOOOOO     SUPERCOMPUTING    " << std::endl <<
    "OOO  OOO  OOOO  B   S   C   OOOO                       " << std::endl <<
    "OOO  OOO  OOOO              OOOO                       " << std::endl <<
    "OOO  OOO  OOOOOOOOOOOOOOOOOOOOOO     CENTER            " << std::endl <<
    " OOO  OO   OOOOOOOOOOOOOOOOOOOO                        " << std::endl <<
    "  OO   OO   OOOOOOOOOOOOOOOOOO                         " << std::endl <<
    "   OOO   OO   OOOOOOOOOOOOOO         CENTRO NACIONAL DE" << std::endl <<
    "     OO    OO    OOOOOOO             SUPERCOMPUTACION  " << std::endl;
}


unsigned long OMPSS_POOL::
Get_n_divisions ()
{
    return this->number_of_divisions;
}

bool OMPSS_POOL::
Set_n_divisions (unsigned long n)
{
    this->number_of_divisions = n;
    return true;
}
