import data from './data.json' assert { type: "json" };
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { calculateScore, showTopMovies } from './common.mjs';
import { methods } from './distance.mjs';

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

if (otherUsersInOrderOfSimilarity.length === 0) {
  // FIXME: Tu dało by się użyć k-mean żeby pogrupować użytkowników gdy ktoś się ze sobą nie pokrywa (było na wykładzie)
  // Można użyć `@seregpie/k-means`
  process.exit(2);
}

console.log('Users similar to you:', otherUsersInOrderOfSimilarity.map(x => `${x.user.name} ${x.score}`));

const mostSimilarRecord = otherUsersInOrderOfSimilarity.at(1);
const leastSimilarRecord = otherUsersInOrderOfSimilarity.at(-1);

console.log('Top 5 movies to watch:');
await showTopMovies(5, mostSimilarRecord.user.ratings, requestedUser.ratings, true);
console.log('Top 5 movies not to watch:');
await showTopMovies(5, leastSimilarRecord.user.ratings, requestedUser.ratings, false);
