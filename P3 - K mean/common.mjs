/**
 * @param a {UserScores}
 * @param b {UserScores}
 * @returns {number[]} Ratings of the user a
 */
export const intersectUserScores = (a, b) => Object
  .entries(a) // For each score
  .filter(([movieId, rating]) => rating && b[movieId]) // Leave only ratings that both users have
  .sort(([movieIdA], [movieIdB]) => movieIdA - movieIdB) // Sort ratings by move id, so comparison will be possible
  .map(([_, rating]) => rating); // Leave only ratings, where index is new move id

/**
 * @param distanceMethod {(a: UserScores, b: UserScores) => number}
 * @param a {UserScores}
 * @param b {UserScores}
 * @returns {number} -1 means that users do not overlap
 */
export const calculateScore = (distanceMethod, a, b) => {
  const ratingsOfUserA = intersectUserScores(a, b);
  const ratingsOfUserB = intersectUserScores(b, a);

  if (ratingsOfUserA.length === 0 || ratingsOfUserB.length === 0) {
    return -1;
  }

  const distanceBetweenUsers = distanceMethod(ratingsOfUserA, ratingsOfUserB);

  return 1 / (1 + distanceBetweenUsers);
};
