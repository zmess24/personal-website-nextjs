// The pixel monogram shared with CLZ Labs: 5x7 glyphs drawn as 1px squares.
const GLYPHS = {
	C: [".####", "#....", "#....", "#....", "#....", "#....", ".####"],
	L: ["#....", "#....", "#....", "#....", "#....", "#....", "#####"],
	Z: ["#####", "....#", "...#.", "..#..", ".#...", "#....", "#####"],
	M: ["#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"],
};

export default function PixelMark({ word, label, className }) {
	const rects = [];
	[...word].forEach((ch, i) => {
		GLYPHS[ch].forEach((row, y) => {
			[...row].forEach((v, x) => {
				if (v === "#") rects.push(<rect key={`${i}-${x}-${y}`} x={i * 6 + x} y={y} width="1" height="1" />);
			});
		});
	});
	return (
		<svg className={className} viewBox={`0 0 ${word.length * 6 - 1} 7`} role="img" aria-label={label} shapeRendering="crispEdges">
			{rects}
		</svg>
	);
}
