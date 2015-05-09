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

#ifndef SERIALBLOCKALIGNERPARAMETERS_HPP_
#define SERIALBLOCKALIGNERPARAMETERS_HPP_

#include "libmasa/libmasa.hpp"

#define ITERATE_BY_ROWS			(0)
#define ITERATE_BY_COLS			(1)
#define ITERATE_BY_SQUARE		(2)
#define ITERATE_BY_SQUARE_INV	(3)
#define ITERATE_BY_DIAGONAL		(4)


class SerialBlockAlignerParameters : public BlockAlignerParameters {
private:
	/** Order of block processing */
	int blockOrder;

public:
	SerialBlockAlignerParameters();
	virtual ~SerialBlockAlignerParameters();
	virtual void printUsage() const;

	/**
	 * @see AbstractAlignerParameters::processArgument() method.
	 */
	int processArgument(int argv, char** argc);

	int getBlockOrder() const;
};

#endif /* SERIALBLOCKALIGNERPARAMETERS_HPP_ */
