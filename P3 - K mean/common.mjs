/**
 * @param a {UserScores}
 * @param b {UserScores}
 * @returns {number[]} Ratings of the user a common with b
 */
export const intersectUserScores = (a, b) => Object
  .entries(a) // For each score
  .filter(([movieId, rating]) => rating && b[movieId]) // Leave only ratings that both users have
  .sort(([movieIdA], [movieIdB]) => movieIdA - movieIdB) // Sort ratings by move id, so comparison will be possible
  .map(([_, rating]) => rating); // Leave only ratings, where index is new move id

/**
 * @param a {UserScores}
 * @param b {UserScores}
 * @param asc {boolean}
 * @returns {string[]} Movies of the user a that b doesn't have
 */
export const bisectUserScores = (a, b, asc) => Object
  .entries(a) // For each score
  .filter(([movieId, rating]) => rating && !b[movieId]) // Leave only ratings that second user doesn't have
  .sort(([,ratingA], [,ratingB]) => (ratingB - ratingA) * (asc ? 1 : -1)) // Sort ratings by score
  .map(([movieId]) => movieId); // Leave only ratings, where index is new move id

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

/**
 * Fetch movie details from the IMDB API.
 * @param {string} movieId - IMDB movie ID
 * @returns {Promise<{
 * sucess: boolean
 * result: Record<string, string> & {
 *   Ratings: {Source: string, Value: string}[]
 * }
 * }>} - Movie details
 */
export const fetchMovieDetails = async (movieId) => {
  try {
    const response = await fetch(`https://api.collectapi.com/imdb/imdbSearchById?movieId=${movieId}`, {
      headers: {
        'authorization': process.env['IMDB_API_KEY'],
        'content-type': 'application/json',
      },
    });

    return await response.json();
  } catch (error) {
    console.error('Error fetching movie details:', error.message);
    return null;
  }
}

/**
 * Show top {count} movies from IMDB API
 * @param count {number} Count of movies to show
 * @param recommendedScores {UserScores} Ratings of the most similar user to target user
 * @param targetUserScores {UserScores} Ratings of th target user (need to filter out movies that he already watched)
 * @param asc {boolean}
 * @return {Promise<void>}
 */
export const showTopMovies = async (count, recommendedScores, targetUserScores, asc) => {
  const exclusiveIds = bisectUserScores(recommendedScores, targetUserScores, asc).splice(0, count);

  console.log(exclusiveIds)

  if (exclusiveIds.length < count) {
    console.log(`We couldn't recommend this many movies, we will show only ${exclusiveIds.length} findings:`);
  }

  const moveDetails = await Promise.all(exclusiveIds.map(id => fetchMovieDetails(id)));

  console.dir(moveDetails.map(x => x.result), {
    depth: 3,
    color: true,
  });
}
