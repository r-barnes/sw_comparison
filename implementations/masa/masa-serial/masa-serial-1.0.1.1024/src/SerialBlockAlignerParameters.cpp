/*******************************************************************************
 *
 * Copyright (c) 2010-2015   Edans Sandes
 *
 * This file is part of MASA-Serial.
 * 
 * MASA-Serial is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * MASA-Serial is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MASA-Serial.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include "SerialBlockAlignerParameters.hpp"

#define DEFAULT_BLOCK_ORDER		ITERATE_BY_DIAGONAL
#define DEFAULT_BLOCK_ORDER_STR	"diagonals"

#include <sstream>
using namespace std;

/**
 * Usage Strings
 */
#define USAGE_HEADER	"Specific Parameters"
#define USAGE "\
--block-order=<type>         Defines the blocking processing order.\n\
                             Possible values: rows, cols, squares, squares-inv, diagonals.\n\
                             Default: "DEFAULT_BLOCK_ORDER_STR".\n\
"



/**
 * getopt arguments
 */
#define ARG_ORDER        0x1007

static struct option long_options[] = {
        {"block-order",      required_argument,      0, ARG_ORDER},
        {0, 0, 0, 0}
    };



SerialBlockAlignerParameters::SerialBlockAlignerParameters() {
	blockOrder = DEFAULT_BLOCK_ORDER;
}

SerialBlockAlignerParameters::~SerialBlockAlignerParameters() {
	// TODO Auto-generated destructor stub
}

void SerialBlockAlignerParameters::printUsage() const {
	BlockAlignerParameters::printUsage(); // Superclass first
	AbstractAlignerParameters::printFormattedUsage(USAGE_HEADER, USAGE);
}

int SerialBlockAlignerParameters::processArgument(int argc, char** argv) {

	int supRet = BlockAlignerParameters::processArgument(argc, argv); // Superclass first
	if (supRet != ARGUMENT_ERROR_NOT_FOUND) {
		return supRet;
	}

	int ret = AbstractAlignerParameters::callGetOpt(argc, argv, long_options);
	switch ( ret ) {
		case ARG_ORDER:
			if (strcmp(optarg, "rows") == 0) {
				blockOrder = ITERATE_BY_ROWS;
			} else if (strcmp(optarg, "cols") == 0) {
				blockOrder = ITERATE_BY_COLS;
			} else if (strcmp(optarg, "squares") == 0) {
				blockOrder = ITERATE_BY_SQUARE;
			} else if (strcmp(optarg, "squares-inv") == 0) {
				blockOrder = ITERATE_BY_SQUARE_INV;
			} else if (strcmp(optarg, "diagonals") == 0) {
				blockOrder = ITERATE_BY_DIAGONAL;
			} else {
				stringstream out;
				out << "Specify a proper order type.";
				setLastError(out.str().c_str());
				return ARGUMENT_ERROR;
			}
			break;
		default:
			return ret;
	}
	return ARGUMENT_OK;
}

int SerialBlockAlignerParameters::getBlockOrder() const {
	return blockOrder;
}
