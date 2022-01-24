#include <iostream>
#include <stdio.h>
#include "clooptools.h"

using namespace std;

int main() {
	ltini();
	cout << B0(1000., 50., 80.) << std::endl;
	ltexi();
}