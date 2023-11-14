/**
* Euclidean distance between two points
* @param p1 {number[]} First point
* @param p2 {number[]} Second point
* @return {number} Distance
*/
const euclideanDistance = (p1, p2) => Math.hypot(...Object.keys(p1).map(k => p2[k] - p1[k]));

/**
 * Manhattan distance between two points
 * @param p1 {number[]} First point
 * @param p2 {number[]} Second point
 * @return {number} Distance
 */
const manhattanDistance = (p1, p2) => Object.keys(p1).map(k => Math.abs(p2[k] - p1[k])).reduce((p, c) => p + c);

export const methods = {
  'euclidean': euclideanDistance,
  'manhattan': manhattanDistance,
};
