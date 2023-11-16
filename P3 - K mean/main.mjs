/**
 * Movie Recommendation System using k-Means Clustering
 *
 * This program implements a movie recommendation system based on k-Means clustering.
 * It takes user input, calculates the similarity between users using a specified distance method,
 * and then recommends movies based on the preferences of similar users.
 *
 * The program reads movie ratings data from 'data.json'.
 * Each user's ratings are represented as a vector of scores [0-10] for various movies.
 *
 * k-Means Clustering:
 * - If there are no users similar to the requested user, the program dynamically determines
 *   the number of clusters based on the total number of users and performs k-Means clustering.
 * - Clusters represent groups of users with similar movie preferences.
 * - You can force usage of k-mean even if there is directly similar user to the requested user, by setting FORCE_K_MEAN to true
 *
 * Usage:
 * - Run the program and input the name of the user for whom you want movie recommendations.
 * - Specify the distance method (e.g., Euclidean, Manhattan) as a command-line argument.
 * - The program will output users similar to the requested user and recommend movies to watch
 *   and movies to avoid.
 *
 * Author: Mateusz Budzisz i Aleksander Guzik
 */
import data from './data.json' assert { type: "json" };
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { calculateScore, showTopMovies } from './common.mjs';
import { methods } from './distance.mjs';
import KMeans from '@seregpie/k-means';

const FORCE_K_MEAN = false; // Set to true if you want to use k-mean even if there is directly similar user to requested user

/**
 * @typedef Unit Represents structure from data.json
 * @extends {Record<string, number>}
 * @property {string} name
 */

/**
 * @typedef UserScores Values [0-10], when the score is less than 1 that means the user has not rated the movie
 * @extends {Record<string, number>}
 */

/**
 * @typedef User
 * @property {string} name
 * @property {UserScores} ratings
 */

/** @type Unit[] Type assertion for IDE completion */
const units =  data;
/** @type {Map<User["name"], User>} Faster lookup by name, separate score from name */
const usersMap = new Map(units.map(x => [x.name.toLocaleLowerCase(), { name: x.name, ratings: Object.fromEntries(Object.entries(x).filter(([k, v]) => k !== 'name' && v)) }]));

/** @see {User.name} is value of user, so we need to subtract one from the count of the values */
console.log(`We have information about ${units.length} users and ${Object.values(data[0]).length - 1} movies total.`);

// Create asynchronous prompt interface
const rl = readline.createInterface({ input, output });

// Ask for name
let requestedName = '';
do {
  requestedName = await rl
    .question('Type the name of the user you want to recommend and discommend movies: ')
    .then(x => x.toLocaleLowerCase());

  if (usersMap.has(requestedName)) break;
  console.warn('We do not have information about this user :c');
} while (true);

// Close stdin, still waiting for native using statement :/
rl.close();

// Set distance method based on arguments
const distanceMethod = methods[process.argv[2]];

if (!distanceMethod) {
  console.error(`We do not support "${process.argv[2]}" distance method`);
  process.exit(1);
}

const requestedUser = usersMap.get(requestedName);

const otherUsersInOrderOfSimilarity = Array
  .from(usersMap.values())
  .filter(user => user.name !== requestedUser.name) // Remove requested users from recommendations
  .map(/** @type User */ user => ({ user, score: calculateScore(distanceMethod, user.ratings, requestedUser.ratings)}))
  .filter(({ user, score }) => score > 0) // Leave users that have at least one common movie with the requested user
  .sort((a, b) => b.score - a.score);

/** @type UserScores */
let mostSimilarUserRatings;
/** @type UserScores */
let leastSimilarUserRatings;

// When there is no direct comparable user, use k-mean clustering
if (otherUsersInOrderOfSimilarity.length === 0 || FORCE_K_MEAN) {
  /** @type User[][] */
  const clusters = KMeans(usersMap.values(), 2, { // 2 Clusters, so one will contain target user and second the opposite of user
    map: (/** @type User */ user) => Object
      .entries(user.ratings)
      .sort(([movieIdA], movieIdB) => movieIdB - movieIdA)
      .map(([, ratings]) => ratings),
    distance: distanceMethod,
    maxIterations: 1, // Disable high resolution of the result, due to poor data :)
  });

  const usersMostSimilarToTargetUser = clusters.find(x => x.some(u => u.name === requestedUser.name));
  const usersLeastSimilarToTargetUser = clusters.find(x => x.every(u => u.name !== requestedUser.name));

  if (usersLeastSimilarToTargetUser.length === 0 || usersMostSimilarToTargetUser.length === 0) {
    console.error('Could not determine similarities with target user due to low volume of data');
    process.exit(1)
  }

  console.log('Users similar to you:', usersMostSimilarToTargetUser.map(x => `${x.name} ${calculateScore(distanceMethod, requestedUser.ratings, x.ratings)}`));
  leastSimilarUserRatings = usersMostSimilarToTargetUser.at(0).ratings;
  mostSimilarUserRatings = usersLeastSimilarToTargetUser.at(0).ratings;
} else {
  console.log('Users similar to you:', otherUsersInOrderOfSimilarity.map(x => `${x.user.name} ${x.score}`));

  mostSimilarUserRatings = otherUsersInOrderOfSimilarity.at(1).user.ratings;
  leastSimilarUserRatings = otherUsersInOrderOfSimilarity.at(-1).user.ratings;
}

console.log('Top 5 movies to watch:');
await showTopMovies(5, mostSimilarUserRatings, requestedUser.ratings, true);
console.log('Top 5 movies not to watch:');
await showTopMovies(5, leastSimilarUserRatings, requestedUser.ratings, false);
