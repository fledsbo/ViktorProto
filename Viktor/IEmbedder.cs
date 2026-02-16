namespace Viktor
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;

    internal interface IEmbedder
    {
        int Dimensions { get; }
        float[] Embed(string text);
        float[][] EmbedBatch(IEnumerable<string> texts);

    }
}
