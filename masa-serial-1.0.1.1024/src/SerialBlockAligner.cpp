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

#include "SerialBlockAligner.hpp"
#include "SerialBlockAlignerParameters.hpp"

SerialBlockAligner::SerialBlockAligner(AbstractBlockProcessor* blockProcessor)
		: AbstractBlockAligner(blockProcessor, new SerialBlockAlignerParameters()) {
}

SerialBlockAligner::~SerialBlockAligner() { }

/*
 * Method that schedules the block execution. See the AbstractBlockAligner
 * class documentation for more information.
 */
void SerialBlockAligner::scheduleBlocks(int grid_width, int grid_height) {
	SerialBlockAlignerParameters* params = (SerialBlockAlignerParameters*) (getParameters());
	int blockOrder = params->getBlockOrder();

	/*
	 * block iteration. Note that the iterations must stop as soon as the
	 * mustContinue() method stops. This usually happens in stages 2 and 3.
	 */
	if (blockOrder == ITERATE_BY_ROWS) {
		/* Here we iterate per rows */
		for (int by = 0; (by < grid_height) && mustContinue(); by++) {
			for (int bx = 0; (bx < grid_width) && mustContinue(); bx++) {
				AbstractBlockAligner::alignBlock(bx, by);
			}
		}
	} else 	if (blockOrder == ITERATE_BY_COLS) {
		/* Here we iterate per rows */
		for (int bx = 0; (bx < grid_width) && mustContinue(); bx++) {
			for (int by = 0; (by < grid_height) && mustContinue(); by++) {
				AbstractBlockAligner::alignBlock(bx, by);
			}
		}
	} else if (blockOrder == ITERATE_BY_DIAGONAL) {
		/* Here we iterate per diagonals */
		for (int d = 0; (d < grid_height+grid_width+1) && mustContinue(); d++) {
			for (int bx=0; (bx<=d && bx<grid_width) && mustContinue(); bx++) {
				int by=d-bx;
				if (by >= grid_height) continue;
				AbstractBlockAligner::alignBlock(bx, by);
			}
		}
	} else if (blockOrder == ITERATE_BY_SQUARE) {
		/* Here we iterate per squared waves */
		for (int j = 0; (j<grid_height || j<grid_width) && mustContinue(); j++) {
			for (int i = 0; (i < j) && mustContinue(); i++) {
				if (i<grid_width && j<grid_height) AbstractBlockAligner::alignBlock(i, j);
				if (j<grid_width && i<grid_height) AbstractBlockAligner::alignBlock(j, i);
			}
			if (j<grid_width && j<grid_height) AbstractBlockAligner::alignBlock(j, j);
		}
	} else if (blockOrder == ITERATE_BY_SQUARE_INV) {
		/* Here we iterate per inverted squared waves */
		for (int j = 0; (j < grid_height && j<grid_width) && mustContinue(); j++) {
			AbstractBlockAligner::alignBlock(j, j);
			for (int i = j+1; (i < grid_height || i < grid_width) && mustContinue(); i++) {
				if (i<grid_width) AbstractBlockAligner::alignBlock(i, j);
				if (i<grid_height) AbstractBlockAligner::alignBlock(j, i);
			}
		}
	}
}

/*
 * Method that transfers the cells to and from the block and calls the
 * AbstractBlockAligner::processBlock function.
 */
void SerialBlockAligner::alignBlock(int bx, int by, int i0, int j0, int i1, int j1) {
	/* The blocks in the top-most positions must read the first row */
	if (by == 0) {
		/* Receives the first row chunk [j0..j1] from the MASA-core */
		receiveFirstRow(row[bx], j1 - j0);
	}

	/* The blocks in the left-most positions must read the first column */
	if (bx == 0) {
		/* Puts the H[i-1][j-1] dependency in the first cell of the column. */
		col[by][0] = getFirstColumnTail();

		/* Receives the first column chunk from the MASA-core */
		receiveFirstColumn(col[by]+1, i1 - i0);
	}

	/*
	 * The processBlock function executes the SW/NW recurrence function
	 * for all the cells of this block.
	 * See the input/output parameters description in the function implementation.
	 */
	if (!mustContinue()) return; // MASA-core is telling to stop
	AbstractBlockAligner::processBlock(bx, by, i0, j0, i1, j1);

	/*
	 * Dispatch special rows with a minimum defined distance (getSpecialRowInterval()).
	 */
	if (isSpecialRow(by)) {
		if (bx == 0) {
			/* The left-most blocks must send an extra cell to represent the
			 * very first column of the matrix. */
			cell_t first_cell = getFirstColumnTail();
			first_cell.f = -INF;
			dispatchRow(i1, &first_cell, 1);
		}
		dispatchRow(i1, row[bx], j1-j0); // Dispatch the special row.
	}

	/*
	 * The right-most blocks must dispatch the last column to MASA.
	 */
	if (isSpecialColumn(bx)) {
		if (by == 0) {
			/* The left-most blocks must send an extra cell to represent the
			 * very first column of the matrix. */
			cell_t first_cell = getFirstRowTail();
			first_cell.f = -INF;
			dispatchColumn(j1, &first_cell, 1);
		}
		dispatchColumn(j1, col[by]+1, i1-i0);
	}
}
